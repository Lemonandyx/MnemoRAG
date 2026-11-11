import asyncio
import itertools
import os
import time
from collections import deque
from copy import deepcopy
from typing import List

import numpy as np
from transformers import AutoTokenizer

from mnemorag.generator.utils import resolve_max_tokens


class _AsyncAdmissionPermit:
    def __init__(self, limiter, token_cost):
        self.limiter = limiter
        self.token_cost = token_cost
        self.prefill_released = token_cost == 0
        self.sequence_released = False

    async def release_prefill(self):
        if self.prefill_released:
            return
        async with self.limiter.condition:
            if self.prefill_released:
                return
            self.limiter.prefill_tokens_in_use -= self.token_cost
            self.prefill_released = True
            self.limiter.condition.notify_all()

    async def release(self):
        await self.release_prefill()
        if self.sequence_released:
            return
        async with self.limiter.condition:
            if self.sequence_released:
                return
            self.limiter.active_sequences -= 1
            self.sequence_released = True
            self.limiter.completed_admissions += 1
            self.limiter.condition.notify_all()


class _AsyncGenerationAdmissionLimiter:
    """FIFO application-side guard for requests submitted to AsyncLLM."""

    def __init__(self, max_sequences, max_prefill_tokens):
        self.max_sequences = max(1, int(max_sequences))
        self.max_prefill_tokens = max(0, int(max_prefill_tokens))
        self.condition = asyncio.Condition()
        self.waiters = deque()
        self.active_sequences = 0
        self.prefill_tokens_in_use = 0
        self.max_active_sequences = 0
        self.max_prefill_tokens_in_use = 0
        self.max_waiters = 0
        self.total_admissions = 0
        self.completed_admissions = 0
        self.waited_admissions = 0
        self.total_wait_s = 0.0

    def _fits(self, token_cost):
        sequence_fits = self.active_sequences < self.max_sequences
        token_fits = (
            self.max_prefill_tokens == 0
            or self.prefill_tokens_in_use + token_cost <= self.max_prefill_tokens
        )
        return sequence_fits and token_fits

    async def acquire(self, token_cost):
        token_cost = max(0, int(token_cost))
        if self.max_prefill_tokens:
            token_cost = min(token_cost, self.max_prefill_tokens)
        else:
            token_cost = 0
        ticket = object()
        queued_at = time.perf_counter()
        waited = False
        async with self.condition:
            self.waiters.append((ticket, token_cost))
            self.max_waiters = max(self.max_waiters, len(self.waiters))
            try:
                while self.waiters[0][0] is not ticket or not self._fits(token_cost):
                    waited = True
                    await self.condition.wait()
            except BaseException:
                self.waiters = deque(
                    item for item in self.waiters if item[0] is not ticket
                )
                self.condition.notify_all()
                raise
            self.waiters.popleft()
            wait_s = max(0.0, time.perf_counter() - queued_at)
            self.active_sequences += 1
            self.prefill_tokens_in_use += token_cost
            self.max_active_sequences = max(
                self.max_active_sequences, self.active_sequences
            )
            self.max_prefill_tokens_in_use = max(
                self.max_prefill_tokens_in_use,
                self.prefill_tokens_in_use,
            )
            self.total_admissions += 1
            self.total_wait_s += wait_s
            if waited:
                self.waited_admissions += 1
            self.condition.notify_all()
        return _AsyncAdmissionPermit(self, token_cost)

    def snapshot(self):
        return {
            "policy": "fifo_max_num_seqs_and_prefill_token_budget",
            "prefill_token_release_boundary": (
                "first_generated_token_or_request_completion"
            ),
            "max_num_seqs": self.max_sequences,
            "max_num_batched_tokens": self.max_prefill_tokens,
            "active_sequences": self.active_sequences,
            "prefill_tokens_in_use": self.prefill_tokens_in_use,
            "queued_admissions": len(self.waiters),
            "max_active_sequences": self.max_active_sequences,
            "max_prefill_tokens_in_use": self.max_prefill_tokens_in_use,
            "max_queued_admissions": self.max_waiters,
            "total_admissions": self.total_admissions,
            "completed_admissions": self.completed_admissions,
            "waited_admissions": self.waited_admissions,
            "total_wait_s": self.total_wait_s,
            "permit_leak": bool(
                self.active_sequences or self.prefill_tokens_in_use or self.waiters
            ),
        }


class BaseGenerator:
    """Common configuration shared by generator implementations."""

    def __init__(self, config):
        self._config = config
        self.update_config()

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config_data):
        self._config = config_data
        self.update_config()

    def update_config(self):
        self.update_base_setting()
        self.update_additional_setting()

    def update_base_setting(self):
        self.model_name = self._config["generator_model"]
        self.model_path = self._config["generator_model_path"]

        self.max_input_len = self._config["generator_max_input_len"]
        self.batch_size = self._config["generator_batch_size"]
        self.device = self._config["device"]
        self.gpu_num = self._config["gpu_num"]
        self.generation_params = self._config["generation_params"]

    def update_additional_setting(self):
        raise NotImplementedError

    def generate(self, input_list: list) -> List[str]:
        """Generate one response for each input prompt.

        Args:
            input_list: it contains input texts, each item represents a sample.

        Returns:
            list: contains generator's response of each input sample.
        """
        raise NotImplementedError


class VLLMGenerator(BaseGenerator):
    """Class for decoder-only generator, based on vllm."""

    supports_document_heat_metadata = True

    @staticmethod
    def _persistent_async_enabled(config):
        try:
            configured = bool(config["vllm_persistent_async_engine"])
        except Exception:
            configured = False
        override = os.environ.get("VLLM_PERSISTENT_ASYNC_ENGINE")
        if override is None:
            return configured
        return override == "1"

    def __init__(self, config):
        super().__init__(config)
        self.use_persistent_async_engine = self._persistent_async_enabled(self._config)
        try:
            self.enable_prefix_caching = bool(self._config["enable_prefix_caching"])
        except Exception:
            self.enable_prefix_caching = True
        kv_transfer_config = None
        cpu_kv_blocks = int(os.environ.get("MNEMORAG_CPU_KV_BLOCKS", "0"))
        if cpu_kv_blocks > 0:
            from vllm.config import KVTransferConfig

            cpu_kv_block_size = int(os.environ.get("MNEMORAG_CPU_KV_BLOCK_SIZE", "16"))
            cpu_kv_eviction_policy = os.environ.get(
                "MNEMORAG_CPU_KV_EVICTION_POLICY", "heat"
            )
            kv_transfer_config = KVTransferConfig(
                kv_connector="OffloadingConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "num_cpu_blocks": cpu_kv_blocks,
                    "block_size": cpu_kv_block_size,
                    "eviction_policy": cpu_kv_eviction_policy,
                },
            )
            print(
                f"CPU_KV blocks={cpu_kv_blocks}, "
                f"block_size={cpu_kv_block_size}, "
                f"eviction_policy={cpu_kv_eviction_policy}",
                flush=True,
            )
        if self.use_persistent_async_engine:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM

            self.async_admission_max_num_seqs = int(
                os.environ.get("VLLM_MAX_NUM_SEQS", "256")
            )
            self.async_admission_max_prefill_tokens = int(
                os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "0")
            )
            engine_kwargs = {
                "model": self.model_path,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_model_len": self.max_model_len,
                "disable_log_stats": False,
                "kv_cache_dtype": os.environ.get("VLLM_KV_CACHE_DTYPE", "auto"),
                "calculate_kv_scales": (
                    os.environ.get("VLLM_CALCULATE_KV_SCALES", "0") == "1"
                ),
                "enable_prefix_caching": self.enable_prefix_caching,
                "kv_transfer_config": kv_transfer_config,
                "max_num_seqs": self.async_admission_max_num_seqs,
            }
            max_num_batched_tokens = self.async_admission_max_prefill_tokens
            if max_num_batched_tokens > 0:
                engine_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
            if self.use_lora:
                engine_kwargs.update(enable_lora=True, max_lora_rank=64)
            self.model = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs))
            self.async_request_counter = itertools.count()
            print("Persistent AsyncLLM engine enabled", flush=True)
        elif self.use_lora:
            from vllm import LLM

            self.model = LLM(
                self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enable_lora=True,
                max_lora_rank=64,
                max_logprobs=32016,
                max_model_len=self.max_model_len,
                kv_cache_dtype=os.environ.get("VLLM_KV_CACHE_DTYPE", "auto"),
                calculate_kv_scales=(
                    os.environ.get("VLLM_CALCULATE_KV_SCALES", "0") == "1"
                ),
                enable_prefix_caching=self.enable_prefix_caching,
                kv_transfer_config=kv_transfer_config,
                max_num_seqs=int(os.environ.get("VLLM_MAX_NUM_SEQS", "256")),
            )
        else:
            from vllm import LLM

            self.model = LLM(
                self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_logprobs=32016,
                max_model_len=self.max_model_len,
                disable_log_stats=False,
                kv_cache_dtype=os.environ.get("VLLM_KV_CACHE_DTYPE", "auto"),
                calculate_kv_scales=(
                    os.environ.get("VLLM_CALCULATE_KV_SCALES", "0") == "1"
                ),
                enable_prefix_caching=self.enable_prefix_caching,
                kv_transfer_config=kv_transfer_config,
                max_num_seqs=int(os.environ.get("VLLM_MAX_NUM_SEQS", "256")),
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.exact_prefix_cached_tokens = 0
        self.exact_prefix_prompt_tokens = 0
        self.debug_generation_stats = (
            os.environ.get("MNEMORAG_GENERATION_DEBUG_TIMING", "0") == "1"
        )
        native_first_token_override = os.environ.get(
            "MNEMORAG_NATIVE_FIRST_TOKEN_EVENT"
        )
        self.use_native_first_token_event = bool(
            self.use_persistent_async_engine
            and getattr(self.model, "supports_mnemorag_first_token_event", False)
            and native_first_token_override == "1"
        )
        self.use_native_prefill_start_event = bool(
            self.use_persistent_async_engine
            and getattr(self.model, "supports_mnemorag_prefill_start_event", False)
        )
        self.async_request_count = 0
        self.async_stream_update_count = 0

    def _get_async_admission_limiter(self):
        max_sequences = int(
            getattr(
                self,
                "async_admission_max_num_seqs",
                os.environ.get("VLLM_MAX_NUM_SEQS", "256"),
            )
        )
        max_prefill_tokens = int(
            getattr(
                self,
                "async_admission_max_prefill_tokens",
                os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "0"),
            )
        )
        limiter = getattr(self, "_async_admission_limiter", None)
        if limiter is None:
            limiter = _AsyncGenerationAdmissionLimiter(
                max_sequences, max_prefill_tokens
            )
            self._async_admission_limiter = limiter
        elif (
            limiter.max_sequences != max_sequences
            or limiter.max_prefill_tokens != max_prefill_tokens
        ):
            if limiter.active_sequences or limiter.waiters:
                raise RuntimeError(
                    "Cannot change AsyncLLM admission limits while requests are active"
                )
            limiter = _AsyncGenerationAdmissionLimiter(
                max_sequences, max_prefill_tokens
            )
            self._async_admission_limiter = limiter
        return limiter

    def get_async_admission_stats(self):
        limiter = getattr(self, "_async_admission_limiter", None)
        if limiter is not None:
            return limiter.snapshot()
        return {
            "policy": "fifo_max_num_seqs_and_prefill_token_budget",
            "prefill_token_release_boundary": (
                "first_generated_token_or_request_completion"
            ),
            "max_num_seqs": int(
                getattr(
                    self,
                    "async_admission_max_num_seqs",
                    os.environ.get("VLLM_MAX_NUM_SEQS", "256"),
                )
            ),
            "max_num_batched_tokens": int(
                getattr(
                    self,
                    "async_admission_max_prefill_tokens",
                    os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "0"),
                )
            ),
            "active_sequences": 0,
            "prefill_tokens_in_use": 0,
            "queued_admissions": 0,
            "max_active_sequences": 0,
            "max_prefill_tokens_in_use": 0,
            "max_queued_admissions": 0,
            "total_admissions": 0,
            "completed_admissions": 0,
            "waited_admissions": 0,
            "total_wait_s": 0.0,
            "permit_leak": False,
        }

    def _estimate_admission_tokens(self, request_prompt):
        if isinstance(request_prompt, dict):
            token_ids = request_prompt.get("prompt_token_ids")
            if token_ids is not None:
                return max(1, len(token_ids))
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None and isinstance(request_prompt, str):
            encoded = tokenizer(request_prompt, add_special_tokens=True)
            return max(1, len(encoded["input_ids"]))
        return max(1, (len(str(request_prompt)) + 3) // 4)

    def shutdown(self):
        """Release the persistent engine before interpreter teardown."""
        if not self.use_persistent_async_engine:
            return
        model = getattr(self, "model", None)
        shutdown = getattr(model, "shutdown", None)
        if shutdown is not None:
            shutdown()

    async def get_kv_cache_stats(self):
        """Return MnemoRAG KV metadata counters from a persistent engine."""
        if not self.use_persistent_async_engine:
            return None
        getter = getattr(self.model, "get_mnemorag_kv_cache_stats", None)
        return await getter() if getter is not None else None

    @staticmethod
    def _signal_first_token(first_token_event, timestamp_s=None):
        if first_token_event is None or first_token_event.is_set():
            return
        first_token_event.mnemorag_set_time_s = (
            time.perf_counter() if timestamp_s is None else timestamp_s
        )
        first_token_event.set()

    def _build_sampling_params(self, params, return_scores):
        from vllm import SamplingParams

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        if "do_sample" in generation_params:
            do_sample_flag = generation_params.pop("do_sample")
            if not do_sample_flag:
                generation_params["temperature"] = 0
        generation_params["seed"] = self._config["seed"]
        generation_params = resolve_max_tokens(
            params, generation_params, prioritize_new_tokens=False
        )
        if "stop" in generation_params:
            generation_params["stop"].append("<|eot_id|>")
            generation_params["include_stop_str_in_output"] = True
        else:
            generation_params["stop"] = ["<|eot_id|>"]
        if return_scores and "logprobs" not in generation_params:
            generation_params["logprobs"] = 100
        return SamplingParams(**generation_params)

    def _prepare_document_heat_request(
        self, prompt: str, document_heat_spans, sampling_params
    ):
        """Tokenize once and attach document heat to exact token intervals."""
        if not document_heat_spans:
            return prompt, sampling_params

        encoded = self.tokenizer(
            prompt,
            # Match vLLM's string-prompt tokenization exactly. Passing
            # pretokenized IDs must not silently drop the model's BOS token.
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        token_ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        token_spans = []
        for span in document_heat_spans:
            char_start = int(span["char_start"])
            char_end = int(span["char_end"])
            heat = max(1.0, float(span["heat"]))
            covered = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > start and start < char_end and end > char_start
            ]
            if covered:
                token_spans.append([covered[0], covered[-1] + 1, heat])

        if not token_spans:
            return prompt, sampling_params

        request_params = deepcopy(sampling_params)
        request_params.extra_args = dict(request_params.extra_args or {})
        request_params.extra_args["mnemorag_document_heat_spans"] = token_spans
        # Reuse the exact IDs used for span construction inside vLLM.
        return {"prompt_token_ids": token_ids}, request_params

    def _format_outputs(self, outputs, return_raw_output, return_scores):
        batch_cached_tokens = sum(
            int(getattr(output, "num_cached_tokens", 0) or 0) for output in outputs
        )
        batch_prompt_tokens = sum(
            len(getattr(output, "prompt_token_ids", None) or ()) for output in outputs
        )
        self.exact_prefix_cached_tokens += batch_cached_tokens
        self.exact_prefix_prompt_tokens += batch_prompt_tokens
        exact_rate = (
            100.0 * self.exact_prefix_cached_tokens / self.exact_prefix_prompt_tokens
            if self.exact_prefix_prompt_tokens
            else 0.0
        )
        if getattr(self, "debug_generation_stats", False):
            print(
                f"Prefix cache exact cumulative hit rate: "
                f"{exact_rate:.6f}% ({self.exact_prefix_cached_tokens}/"
                f"{self.exact_prefix_prompt_tokens} prompt tokens)"
            )
        if return_raw_output:
            base_output = outputs
        else:
            base_output = [
                [candidate.text for candidate in output.outputs]
                if len(output.outputs) > 1
                else output.outputs[0].text
                for output in outputs
            ]
        if not return_scores:
            return base_output

        scores = []
        for output in outputs:
            output_scores = []
            for single_output in output.outputs:
                if single_output.logprobs:
                    output_scores.append(
                        [
                            np.exp(list(score_dict.values())[0].logprob)
                            for score_dict in single_output.logprobs
                        ]
                    )
                else:
                    output_scores.append([])
            scores.append(
                output_scores[0] if len(output_scores) == 1 else output_scores
            )
        return base_output, scores

    async def generate_async(
        self,
        input_list: List[str],
        return_raw_output=False,
        return_scores=False,
        return_completion_times=False,
        first_token_event: asyncio.Event | None = None,
        prefill_start_event: asyncio.Event | None = None,
        prefill_start_after_all: bool = False,
        document_heat_spans=None,
        **params,
    ):
        if not self.use_persistent_async_engine:
            self._signal_first_token(prefill_start_event)
            outputs = await asyncio.to_thread(
                self.generate,
                input_list,
                return_raw_output=return_raw_output,
                return_scores=return_scores,
                document_heat_spans=document_heat_spans,
                **params,
            )
            if first_token_event is not None:
                self._signal_first_token(first_token_event)
            return outputs
        if isinstance(input_list, str):
            input_list = [input_list]
        if document_heat_spans is None:
            document_heat_spans = [None] * len(input_list)
        elif len(document_heat_spans) != len(input_list):
            raise ValueError("document_heat_spans must contain one entry per prompt")
        sampling_params = self._build_sampling_params(params, return_scores)
        from vllm.sampling_params import RequestOutputKind

        use_native_first_token_event = bool(
            first_token_event is not None
            and getattr(self, "use_native_first_token_event", False)
        )
        supports_native_first_token_event = bool(
            self.use_persistent_async_engine
            and getattr(
                self.model,
                "supports_mnemorag_first_token_event",
                False,
            )
        )
        sampling_params.output_kind = (
            RequestOutputKind.CUMULATIVE
            if first_token_event is not None and not use_native_first_token_event
            else RequestOutputKind.FINAL_ONLY
        )
        lora_request = None
        if self.use_lora:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest("lora_module", 1, self.lora_path)

        remaining_first_tokens = len(input_list)
        first_token_lock = asyncio.Lock()
        batch_first_token_time_s = None
        remaining_prefill_starts = len(input_list)
        prefill_start_lock = asyncio.Lock()
        batch_prefill_start_time_s = None
        admission_limiter = self._get_async_admission_limiter()

        async def mark_first_token(timestamp_s=None):
            nonlocal remaining_first_tokens, batch_first_token_time_s
            if first_token_event is None:
                return
            async with first_token_lock:
                if timestamp_s is not None:
                    batch_first_token_time_s = max(
                        timestamp_s,
                        batch_first_token_time_s or timestamp_s,
                    )
                remaining_first_tokens -= 1
                if remaining_first_tokens == 0:
                    self._signal_first_token(
                        first_token_event,
                        timestamp_s=batch_first_token_time_s,
                    )

        async def mark_prefill_start(timestamp_s=None):
            nonlocal remaining_prefill_starts, batch_prefill_start_time_s
            if prefill_start_event is None:
                return
            async with prefill_start_lock:
                if timestamp_s is not None:
                    batch_prefill_start_time_s = max(
                        timestamp_s,
                        batch_prefill_start_time_s or timestamp_s,
                    )
                remaining_prefill_starts -= 1
                should_signal = (
                    remaining_prefill_starts == 0
                    if prefill_start_after_all
                    else not prefill_start_event.is_set()
                )
                if should_signal:
                    self._signal_first_token(
                        prefill_start_event,
                        timestamp_s=batch_prefill_start_time_s,
                    )

        async def collect_output(prompt, prompt_document_heat_spans):
            self.async_request_count = getattr(self, "async_request_count", 0) + 1
            output = None
            received_first_token = False
            request_id = str(next(self.async_request_counter))
            request_prompt, request_sampling_params = (
                self._prepare_document_heat_request(
                    prompt, prompt_document_heat_spans, sampling_params
                )
            )
            token_cost = (
                self._estimate_admission_tokens(request_prompt)
                if admission_limiter.max_prefill_tokens
                else 0
            )
            admission_permit = await admission_limiter.acquire(token_cost)
            request_first_token_event = (
                asyncio.Event()
                if (
                    use_native_first_token_event
                    or (
                        admission_permit.token_cost
                        and supports_native_first_token_event
                    )
                )
                else None
            )
            request_prefill_start_event = (
                asyncio.Event()
                if self.use_native_prefill_start_event
                and prefill_start_event is not None
                else None
            )
            observer = None
            prefill_observer = None
            if request_first_token_event is not None:

                async def observe_first_token():
                    await request_first_token_event.wait()
                    await admission_permit.release_prefill()
                    if use_native_first_token_event:
                        await mark_first_token(
                            getattr(
                                request_first_token_event,
                                "mnemorag_set_time_s",
                                time.perf_counter(),
                            )
                        )

                observer = asyncio.create_task(observe_first_token())

            if request_prefill_start_event is not None:

                async def observe_prefill_start():
                    await request_prefill_start_event.wait()
                    await mark_prefill_start(
                        getattr(
                            request_prefill_start_event,
                            "mnemorag_set_time_s",
                            time.perf_counter(),
                        )
                    )

                prefill_observer = asyncio.create_task(observe_prefill_start())
            else:
                # Engines without a native admission event cannot expose a
                # more precise request-admission boundary. The prefill-token
                # permit is still held until a generated token is observed.
                if prefill_start_event is not None:
                    await mark_prefill_start(time.perf_counter())

            generate_kwargs = {"lora_request": lora_request}
            if request_first_token_event is not None:
                generate_kwargs["first_token_event"] = request_first_token_event
            if request_prefill_start_event is not None:
                generate_kwargs["prefill_start_event"] = request_prefill_start_event
            try:
                async for candidate in self.model.generate(
                    request_prompt,
                    request_sampling_params,
                    request_id,
                    **generate_kwargs,
                ):
                    self.async_stream_update_count = (
                        getattr(self, "async_stream_update_count", 0) + 1
                    )
                    output = candidate
                    if not use_native_first_token_event:
                        has_generated_token = any(
                            bool(getattr(single_output, "token_ids", ()))
                            for single_output in getattr(candidate, "outputs", ())
                        )
                        if not received_first_token and has_generated_token:
                            received_first_token = True
                            await admission_permit.release_prefill()
                            if not use_native_first_token_event:
                                await mark_first_token()
                if observer is not None:
                    if not request_first_token_event.is_set():
                        request_first_token_event.mnemorag_set_time_s = (
                            time.perf_counter()
                        )
                        request_first_token_event.set()
                    await observer
                if prefill_observer is not None:
                    if not request_prefill_start_event.is_set():
                        request_prefill_start_event.mnemorag_set_time_s = (
                            time.perf_counter()
                        )
                        request_prefill_start_event.set()
                    await prefill_observer
                if output is None:
                    raise RuntimeError(f"AsyncLLM returned no output for {request_id}")
                return output, time.perf_counter()
            finally:
                for task in (observer, prefill_observer):
                    if task is not None and not task.done():
                        task.cancel()
                await asyncio.gather(
                    *(task for task in (observer, prefill_observer) if task),
                    return_exceptions=True,
                )
                await admission_permit.release()

        started = time.perf_counter()
        request_tasks = [
            asyncio.create_task(collect_output(prompt, heat_spans))
            for prompt, heat_spans in zip(input_list, document_heat_spans)
        ]
        try:
            completed_outputs = await asyncio.gather(*request_tasks)
        except BaseException:
            for task in request_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*request_tasks, return_exceptions=True)
            raise
        if getattr(self, "debug_generation_stats", False):
            print(
                f"async_generate_outputs time: "
                f"{time.perf_counter() - started:.4f} seconds"
            )
        outputs = [item[0] for item in completed_outputs]
        completion_times = [float(item[1]) for item in completed_outputs]
        formatted_outputs = self._format_outputs(
            outputs, return_raw_output, return_scores
        )
        if return_completion_times:
            return formatted_outputs, completion_times
        return formatted_outputs

    def update_additional_setting(self):
        if "gpu_memory_utilization" not in self._config:
            self.gpu_memory_utilization = 0.85
        else:
            self.gpu_memory_utilization = self._config["gpu_memory_utilization"]
        self.gpu_memory_utilization = float(
            os.environ.get(
                "VLLM_GPU_MEMORY_UTILIZATION",
                self.gpu_memory_utilization,
            )
        )
        if self.gpu_num != 1 and self.gpu_num % 2 != 0:
            self.tensor_parallel_size = self.gpu_num - 1
        else:
            self.tensor_parallel_size = self.gpu_num

        self.lora_path = (
            None
            if "generator_lora_path" not in self._config
            else self._config["generator_lora_path"]
        )
        self.use_lora = False
        if self.lora_path is not None:
            self.use_lora = True
        self.max_model_len = self._config["generator_max_input_len"]

    def generate(
        self,
        input_list: List[str],
        return_raw_output=False,
        return_scores=False,
        document_heat_spans=None,
        **params,
    ):
        if self.use_persistent_async_engine:
            raise RuntimeError(
                "Persistent AsyncLLM requires awaiting VLLMGenerator.generate_async()"
            )
        if isinstance(input_list, str):
            input_list = [input_list]
        if document_heat_spans is None:
            document_heat_spans = [None] * len(input_list)
        elif len(document_heat_spans) != len(input_list):
            raise ValueError("document_heat_spans must contain one entry per prompt")
        sampling_params = self._build_sampling_params(params, return_scores)
        prepared = [
            self._prepare_document_heat_request(prompt, spans, sampling_params)
            for prompt, spans in zip(input_list, document_heat_spans)
        ]
        prepared_inputs = [item[0] for item in prepared]
        prepared_sampling_params = [item[1] for item in prepared]
        has_per_request_metadata = any(document_heat_spans)
        model_sampling_params = (
            prepared_sampling_params if has_per_request_metadata else sampling_params
        )

        if self.use_lora:
            from vllm.lora.request import LoRARequest

            outputs = self.model.generate(
                prepared_inputs,
                model_sampling_params,
                lora_request=LoRARequest("lora_module", 1, self.lora_path),
            )
        else:
            generate_outputs_time_st = time.time()
            outputs = self.model.generate(prepared_inputs, model_sampling_params)
            generate_outputs_time_ed = time.time()
            print(
                f"generate_outputs time: {generate_outputs_time_ed - generate_outputs_time_st:.4f} seconds"
            )
        return self._format_outputs(outputs, return_raw_output, return_scores)
