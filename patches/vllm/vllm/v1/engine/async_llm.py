# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import socket
import time
from collections.abc import AsyncGenerator, Iterable, Mapping
from copy import copy
from typing import Any, cast

import numpy as np
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.utils import _validate_truncation_size
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tracing import init_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.transformers_utils.tokenizer import AnyTokenizer, init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.utils.async_utils import cancel_task_threadsafe
from vllm.utils.collection_utils import as_list
from vllm.utils.math_utils import cdiv
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.engine.processor import Processor
from vllm.v1.executor import Executor
from vllm.v1.metrics.loggers import (
    StatLoggerFactory,
    StatLoggerManager,
    load_stat_logger_plugin_factories,
)
from vllm.v1.metrics.prometheus import shutdown_prometheus
from vllm.v1.metrics.stats import IterationStats

logger = init_logger(__name__)


class AsyncLLM(EngineClient):
    supports_mnemorag_first_token_event = True
    supports_mnemorag_prefill_start_event = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: list[StatLoggerFactory] | None = None,
        aggregate_engine_logging: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> None:
        """Create an asynchronous engine client and its output processor."""
        maybe_register_config_serialize_by_value()

        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.observability_config = vllm_config.observability_config
        self.log_requests = log_requests

        custom_stat_loggers = list(stat_loggers or [])
        custom_stat_loggers.extend(load_stat_logger_plugin_factories())

        has_custom_loggers = bool(custom_stat_loggers)
        self.log_stats = log_stats or has_custom_loggers
        if not log_stats and has_custom_loggers:
            logger.info(
                "AsyncLLM created with log_stats=False, "
                "but custom stat loggers were found; "
                "enabling logging without default stat loggers."
            )

        if self.model_config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = init_tokenizer_from_configs(self.model_config)

        self.processor = Processor(self.vllm_config, tokenizer)
        self.io_processor = get_io_processor(
            self.vllm_config,
            self.model_config.io_processor_plugin,
        )
        self._mnemorag_first_token_events: dict[str, asyncio.Event] = {}
        self._mnemorag_prefill_start_events: dict[str, asyncio.Event] = {}

        stream_interval = self.vllm_config.scheduler_config.stream_interval
        self.output_processor = OutputProcessor(
            self.tokenizer, log_stats=self.log_stats, stream_interval=stream_interval
        )
        self.output_processor.mnemorag_first_token_events = (
            self._mnemorag_first_token_events
        )
        self.output_processor.mnemorag_prefill_start_events = (
            self._mnemorag_prefill_start_events
        )
        endpoint = self.observability_config.otlp_traces_endpoint
        if endpoint is not None:
            tracer = init_tracer("vllm.llm_engine", endpoint)
            self.output_processor.tracer = tracer

        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                engine_idxs=self.engine_core.engine_ranks_managed,
                custom_stat_loggers=custom_stat_loggers,
                enable_default_loggers=log_stats,
                client_count=client_count,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            self.logger_manager.log_engine_initialized()

        self._pause_cond = asyncio.Condition()
        self._paused = False

        self.output_handler: asyncio.Task | None = None
        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            asyncio.get_running_loop()
            self._run_output_handler()
        except RuntimeError:
            pass

        if (
            envs.VLLM_TORCH_PROFILER_DIR
            and not envs.VLLM_TORCH_PROFILER_DISABLE_ASYNC_LLM
        ):
            logger.info(
                "Torch profiler enabled. AsyncLLM CPU traces will be collected under %s",  # noqa: E501
                envs.VLLM_TORCH_PROFILER_DIR,
            )
            if envs.VLLM_PROFILER_MAX_ITERS > 0 or envs.VLLM_PROFILER_DELAY_ITERS > 0:
                logger.warning_once(
                    "Torch profiler received max_iters or delay_iters setting. These "
                    "are not compatible with the AsyncLLM profiler and will be ignored "
                    "for the AsyncLLM process. Engine process profiling will still "
                    "respect these settings. Consider setting "
                    "VLLM_TORCH_PROFILER_DISABLE_ASYNC_LLM=1 to disable "
                    "AsyncLLM profiling."
                )
            worker_name = f"{socket.gethostname()}_{os.getpid()}.async_llm"
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    envs.VLLM_TORCH_PROFILER_DIR, worker_name=worker_name, use_gzip=True
                ),
            )
        else:
            self.profiler = None

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_log_requests: bool = False,
        aggregate_engine_logging: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncLLM":
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
            aggregate_engine_logging=aggregate_engine_logging,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
    ) -> "AsyncLLM":
        """Create an AsyncLLM from engine arguments."""

        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Stop background processes and close IPC resources."""

        shutdown_prometheus()

        if engine_core := getattr(self, "engine_core", None):
            engine_core.shutdown()

        handler = getattr(self, "output_handler", None)
        if handler is not None:
            cancel_task_threadsafe(handler)

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return await self.engine_core.get_supported_tasks_async()

    async def get_mnemorag_kv_cache_stats(self) -> dict[str, Any] | None:
        return await self.engine_core.call_utility_async(
            "get_mnemorag_kv_cache_stats"
        )

    async def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        prompt_text: str | None = None,
    ) -> RequestOutputCollector:
        """Add a request to the asynchronous engine."""

        if self.errored:
            raise EngineDeadError()

        is_pooling = isinstance(params, PoolingParams)

        queue = RequestOutputCollector(output_kind=params.output_kind)

        if isinstance(prompt, EngineCoreRequest):
            request = prompt
        else:
            assert prompt_text is None
            logger.warning_once(
                "Processor has been moved under OpenAIServing and will "
                "be removed from AsyncLLM in v0.13."
            )
            request = self.processor.process_inputs(
                request_id,
                prompt,
                params,
                arrival_time,
                lora_request,
                tokenization_kwargs,
                trace_headers,
                priority,
                data_parallel_rank,
            )
            if isinstance(prompt, str):
                prompt_text = prompt
            elif isinstance(prompt, Mapping):
                prompt_text = cast(str | None, prompt.get("prompt"))

        # Use cloned params that may have been updated in process_inputs()
        params = request.params

        if is_pooling or params.n == 1:
            await self._add_request(request, prompt_text, None, 0, queue)
            return queue

        parent_params = params
        assert isinstance(parent_params, SamplingParams)

        parent_request = ParentRequest(request_id, parent_params)
        for idx in range(parent_params.n):
            request_id, child_params = parent_request.get_child_info(idx)
            child_request = request if idx == parent_params.n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = child_params
            await self._add_request(
                child_request, prompt_text, parent_request, idx, queue
            )
        return queue

    async def _add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        index: int,
        queue: RequestOutputCollector,
    ):
        self.output_processor.add_request(request, prompt, parent_req, index, queue)

        await self.engine_core.add_request_async(request)

        if self.log_requests:
            logger.info("Added request %s.", request.request_id)

    async def generate(
        self,
        prompt: EngineCoreRequest | PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        first_token_event: asyncio.Event | None = None,
        prefill_start_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Submit a generation request and stream its outputs."""

        if (
            self.vllm_config.cache_config.kv_sharing_fast_prefill
            and sampling_params.prompt_logprobs
        ):
            raise ValueError(
                "--kv-sharing-fast-prefill produces incorrect logprobs for "
                "prompt tokens, please disable it when the requests need "
                "prompt logprobs"
            )

        if first_token_event is not None:
            if sampling_params.n != 1:
                raise ValueError(
                    "first_token_event currently requires sampling_params.n == 1"
                )
            self._mnemorag_first_token_events[request_id] = first_token_event
        if prefill_start_event is not None:
            self._mnemorag_prefill_start_events[request_id] = prefill_start_event

        try:
            # Start output handling lazily so construction can precede the event loop.
            self._run_output_handler()

            async with self._pause_cond:
                await self._pause_cond.wait_for(lambda: not self._paused)

            if tokenization_kwargs is None:
                tokenization_kwargs = {}
                truncate_prompt_tokens = sampling_params.truncate_prompt_tokens

                _validate_truncation_size(
                    self.model_config.max_model_len,
                    truncate_prompt_tokens,
                    tokenization_kwargs,
                )

            q = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
                prompt_text=prompt_text,
            )
            # Read outputs delivered by the background handler.
            finished = False
            while not finished:
                # Drain queued outputs without yielding to the event loop.
                out = q.get_nowait() or await q.get()

                finished = out.finished
                assert isinstance(out, RequestOutput)
                yield out

        # Abort if the caller disconnects or cancels the generator.
        except (asyncio.CancelledError, GeneratorExit):
            await self.abort(request_id)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        except ValueError:
            if self.log_requests:
                logger.info("Request %s failed (bad request).", request_id)
            raise

        except Exception as e:
            await self.abort(request_id)
            if self.log_requests:
                logger.info("Request %s failed.", request_id)
            raise EngineGenerateError() from e
        finally:
            self._mnemorag_first_token_events.pop(request_id, None)
            self._mnemorag_prefill_start_events.pop(request_id, None)

    def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        if self.output_handler is not None:
            return

        # Avoid a reference cycle so AsyncLLM can be collected and shut down.
        engine_core = self.engine_core
        output_processor = self.output_processor
        log_stats = self.log_stats
        logger_manager = self.logger_manager
        processor = self.processor

        async def output_handler():
            try:
                while True:
                    outputs = await engine_core.get_output_async()
                    num_outputs = len(outputs.outputs)

                    iteration_stats = (
                        IterationStats() if (log_stats and num_outputs) else None
                    )

                    # Chunk output processing to keep the event loop responsive.
                    if num_outputs <= envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE:
                        slices = (outputs.outputs,)
                    else:
                        slices = np.array_split(
                            outputs.outputs,
                            cdiv(num_outputs, envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE),
                        )

                    for i, outputs_slice in enumerate(slices):
                        processed_outputs = output_processor.process_outputs(
                            outputs_slice, outputs.timestamp, iteration_stats
                        )
                        assert not processed_outputs.request_outputs

                        if i + 1 < len(slices):
                            await asyncio.sleep(0)

                        await engine_core.abort_requests_async(
                            processed_outputs.reqs_to_abort
                        )

                    output_processor.update_scheduler_stats(outputs.scheduler_stats)

                    if logger_manager:
                        logger_manager.record(
                            engine_idx=outputs.engine_index,
                            scheduler_stats=outputs.scheduler_stats,
                            iteration_stats=iteration_stats,
                            mm_cache_stats=processor.stat_mm_cache(),
                        )
            except Exception as e:
                logger.exception("AsyncLLM output_handler failed.")
                output_processor.propagate_error(e)

        self.output_handler = asyncio.create_task(output_handler())

    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort RequestId in OutputProcessor and EngineCore."""

        request_ids = (
            (request_id,) if isinstance(request_id, str) else as_list(request_id)
        )
        all_request_ids = self.output_processor.abort_requests(request_ids)
        await self.engine_core.abort_requests_async(all_request_ids)

        if self.log_requests:
            logger.info("Aborted request(s) %s.", ",".join(request_ids))

    async def pause_generation(
        self,
        *,
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        """Pause request admission, optionally draining requests and clearing caches.

        If wait_for_inflight_requests is false, active requests are aborted.
        """

        async with self._pause_cond:
            if self._paused:
                return
            self._paused = True

        if not wait_for_inflight_requests:
            request_ids = list(self.output_processor.request_states.keys())
            if request_ids:
                await self.abort(request_ids)

        # Wait for running requests to drain before clearing cache.
        if self.output_processor.has_unfinished_requests():
            await self.output_processor.wait_for_requests_to_drain()

        if clear_cache:
            await self.reset_prefix_cache()
            await self.reset_mm_cache()

    async def resume_generation(self) -> None:
        """Resume generation after :meth:`pause_generation`."""

        async with self._pause_cond:
            self._paused = False
            self._pause_cond.notify_all()  # Wake up all waiting requests

    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""

        async with self._pause_cond:
            return self._paused

    async def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        truncate_prompt_tokens: int | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Submit an encoding request and stream its outputs."""

        try:
            # Start output handling lazily so construction can precede the event loop.
            self._run_output_handler()

            async with self._pause_cond:
                await self._pause_cond.wait_for(lambda: not self._paused)

            if tokenization_kwargs is None:
                tokenization_kwargs = {}
            _validate_truncation_size(
                self.model_config.max_model_len,
                truncate_prompt_tokens,
                tokenization_kwargs,
            )

            q = await self.add_request(
                request_id,
                prompt,
                pooling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
            )

            finished = False
            while not finished:
                out = q.get_nowait() or await q.get()
                assert isinstance(out, PoolingRequestOutput)
                finished = out.finished
                yield out

        # Abort the request if the caller disconnects or cancels.
        except asyncio.CancelledError:
            await self.abort(request_id)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        except ValueError:
            if self.log_requests:
                logger.info("Request %s failed (bad request).", request_id)
            raise

        except Exception as e:
            await self.abort(request_id)
            if self.log_requests:
                logger.info("Request %s failed.", request_id)
            raise EngineGenerateError() from e

    @property
    def tokenizer(self) -> AnyTokenizer | None:
        return self.processor.tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: AnyTokenizer | None) -> None:
        self.processor.tokenizer = tokenizer

    async def get_tokenizer(self) -> AnyTokenizer:
        if self.tokenizer is None:
            raise ValueError(
                "Unable to get tokenizer because skip_tokenizer_init is True"
            )

        return self.tokenizer

    async def is_tracing_enabled(self) -> bool:
        return self.observability_config.otlp_traces_endpoint is not None  # type: ignore

    async def do_log_stats(self) -> None:
        if self.logger_manager:
            self.logger_manager.log()

    async def check_health(self) -> None:
        logger.debug("Called check_health.")
        if self.errored:
            raise self.dead_error

    async def start_profile(self) -> None:
        coros = [self.engine_core.profile_async(True)]
        if self.profiler is not None:
            coros.append(asyncio.to_thread(self.profiler.start))
        await asyncio.gather(*coros)

    async def stop_profile(self) -> None:
        coros = [self.engine_core.profile_async(False)]
        if self.profiler is not None:
            coros.append(asyncio.to_thread(self.profiler.stop))
        await asyncio.gather(*coros)

    async def reset_mm_cache(self) -> None:
        self.processor.clear_mm_cache()
        await self.engine_core.reset_mm_cache_async()

    async def reset_prefix_cache(self) -> None:
        await self.engine_core.reset_prefix_cache_async()

    async def sleep(self, level: int = 1) -> None:
        await self.reset_prefix_cache()
        await self.engine_core.sleep_async(level)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(1, level)

    async def wake_up(self, tags: list[str] | None = None) -> None:
        await self.engine_core.wake_up_async(tags)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    async def is_sleeping(self) -> bool:
        return await self.engine_core.is_sleeping_async()

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return await self.engine_core.add_lora_async(lora_request)

    async def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return await self.engine_core.remove_lora_async(lora_id)

    async def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return await self.engine_core.list_loras_async()

    async def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        return await self.engine_core.pin_lora_async(lora_id)

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        """Perform a collective RPC call."""
        return await self.engine_core.collective_rpc_async(
            method, timeout, args, kwargs
        )

    async def wait_for_requests_to_drain(self, drain_timeout: int = 300):
        """Wait for all requests to be drained."""
        start_time = time.time()
        while time.time() - start_time < drain_timeout:
            if not self.engine_core.dp_engines_running():
                logger.info("Engines are idle, requests have been drained")
                return

            logger.info("Engines are still running, waiting for requests to drain...")
            await asyncio.sleep(1)  # Wait 1 second before checking again

        raise TimeoutError(
            f"Timeout reached after {drain_timeout} seconds "
            "waiting for requests to drain."
        )

    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int = 300
    ):
        """Change the data-parallel worker count after draining active requests."""
        old_data_parallel_size = self.vllm_config.parallel_config.data_parallel_size
        if old_data_parallel_size == new_data_parallel_size:
            logger.info(
                "Data parallel size is already %s, skipping scale",
                new_data_parallel_size,
            )
            return
        logger.info(
            "Waiting for requests to drain before scaling up to %s engines...",
            new_data_parallel_size,
        )
        await self.wait_for_requests_to_drain(drain_timeout)
        logger.info(
            "Requests have been drained, proceeding with scale to %s engines",
            new_data_parallel_size,
        )
        await self.engine_core.scale_elastic_ep(new_data_parallel_size)
        self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size

        # Recreating stat loggers resets Prometheus metrics.
        if new_data_parallel_size > old_data_parallel_size and self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=self.vllm_config,
                engine_idxs=list(range(new_data_parallel_size)),
                custom_stat_loggers=None,
            )

    @property
    def is_running(self) -> bool:
        return self.output_handler is None or not self.output_handler.done()

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def errored(self) -> bool:
        return self.engine_core.resources.engine_dead or not self.is_running

    @property
    def dead_error(self) -> BaseException:
        return EngineDeadError()
