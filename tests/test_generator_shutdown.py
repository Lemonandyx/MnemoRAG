import asyncio
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mnemorag.generator.generator import VLLMGenerator


class GeneratorShutdownTest(unittest.TestCase):
    def test_persistent_engine_uses_config_by_default(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(
                VLLMGenerator._persistent_async_enabled(
                    {"vllm_persistent_async_engine": True}
                )
            )
            self.assertFalse(
                VLLMGenerator._persistent_async_enabled(
                    {"vllm_persistent_async_engine": False}
                )
            )

    def test_persistent_engine_environment_override_wins(self):
        config = {"vllm_persistent_async_engine": True}
        with patch.dict(
            "os.environ", {"VLLM_PERSISTENT_ASYNC_ENGINE": "0"}, clear=True
        ):
            self.assertFalse(VLLMGenerator._persistent_async_enabled(config))
        with patch.dict(
            "os.environ", {"VLLM_PERSISTENT_ASYNC_ENGINE": "1"}, clear=True
        ):
            self.assertTrue(
                VLLMGenerator._persistent_async_enabled(
                    {"vllm_persistent_async_engine": False}
                )
            )

    def test_persistent_engine_is_shutdown_explicitly(self):
        generator = object.__new__(VLLMGenerator)
        generator.use_persistent_async_engine = True
        generator.model = Mock()

        generator.shutdown()

        generator.model.shutdown.assert_called_once_with()

    def test_nonpersistent_generator_does_not_shutdown_model(self):
        generator = object.__new__(VLLMGenerator)
        generator.use_persistent_async_engine = False
        generator.model = Mock()

        generator.shutdown()

        generator.model.shutdown.assert_not_called()


class FakeRequestOutput:
    def __init__(self, token_ids=None, text=""):
        self.outputs = [SimpleNamespace(token_ids=list(token_ids or ()), text=text)]

    def add(self, next_output, aggregate):
        assert aggregate is True
        self.outputs[0].token_ids.extend(next_output.outputs[0].token_ids)
        self.outputs[0].text += next_output.outputs[0].text


class FakeStreamingModel:
    def __init__(self, first_token_event):
        self.first_token_event = first_token_event
        self.empty_update_was_ignored = False
        self.token_update_opened_gate = False

    async def generate(self, *_args, **_kwargs):
        yield FakeRequestOutput()
        await asyncio.sleep(0)
        self.empty_update_was_ignored = not self.first_token_event.is_set()
        yield FakeRequestOutput([7], "first")
        yield FakeRequestOutput([7, 8], "first second")
        await asyncio.sleep(0)
        self.token_update_opened_gate = self.first_token_event.is_set()


class StaggeredStreamingModel:
    def __init__(self, fast_ready, release_slow):
        self.fast_ready = fast_ready
        self.release_slow = release_slow

    async def generate(self, prompt, *_args, **_kwargs):
        if prompt == "slow":
            await self.release_slow.wait()
        yield FakeRequestOutput([1], prompt)
        if prompt == "fast":
            self.fast_ready.set()


class NativeFirstTokenModel:
    supports_mnemorag_first_token_event = True

    def __init__(self, first_token_ready, release_final):
        self.first_token_ready = first_token_ready
        self.release_final = release_final
        self.output_kind = None

    async def generate(
        self,
        _prompt,
        sampling_params,
        _request_id,
        *,
        first_token_event,
        **_kwargs,
    ):
        self.output_kind = sampling_params.output_kind
        first_token_event.mnemorag_set_time_s = 123.0
        first_token_event.set()
        self.first_token_ready.set()
        await self.release_final.wait()
        yield FakeRequestOutput([1, 2], "final")


class StaggeredPrefillModel:
    supports_mnemorag_prefill_start_event = True

    def __init__(self, fast_ready, release_slow):
        self.fast_ready = fast_ready
        self.release_slow = release_slow

    async def generate(
        self,
        prompt,
        _sampling_params,
        _request_id,
        *,
        prefill_start_event,
        **_kwargs,
    ):
        if prompt == "slow":
            await self.release_slow.wait()
        prefill_start_event.mnemorag_set_time_s = 10.0 if prompt == "fast" else 20.0
        prefill_start_event.set()
        if prompt == "fast":
            self.fast_ready.set()
        yield FakeRequestOutput([1], prompt)


class PartialFailureModel:
    def __init__(self, sibling_started, sibling_cancelled):
        self.sibling_started = sibling_started
        self.sibling_cancelled = sibling_cancelled
        self.block_sibling = asyncio.Event()

    async def generate(self, prompt, *_args, **_kwargs):
        if prompt == "fail":
            await self.sibling_started.wait()
            raise RuntimeError("injected request failure")
        self.sibling_started.set()
        try:
            await self.block_sibling.wait()
        finally:
            self.sibling_cancelled.set()
        yield FakeRequestOutput([1], "unreachable")


class AdmissionBoundModel:
    supports_mnemorag_prefill_start_event = True

    def __init__(self, two_started, release):
        self.two_started = two_started
        self.release = release
        self.started = []

    async def generate(
        self,
        prompt,
        _sampling_params,
        _request_id,
        *,
        prefill_start_event=None,
        **_kwargs,
    ):
        self.started.append(prompt)
        if prefill_start_event is not None:
            prefill_start_event.mnemorag_set_time_s = 1.0
            prefill_start_event.set()
        if len(self.started) == 2:
            self.two_started.set()
        await self.release.wait()
        yield FakeRequestOutput([1], prompt)


class TokenBudgetModel:
    supports_mnemorag_first_token_event = True

    def __init__(
        self,
        first_started,
        release_first_token,
        all_started,
        release_final,
    ):
        self.first_started = first_started
        self.release_first_token = release_first_token
        self.all_started = all_started
        self.release_final = release_final
        self.started = []

    async def generate(
        self,
        prompt,
        _sampling_params,
        _request_id,
        *,
        first_token_event,
        **_kwargs,
    ):
        self.started.append(prompt)
        if len(self.started) == 1:
            self.first_started.set()
            await self.release_first_token.wait()
        first_token_event.mnemorag_set_time_s = 2.0
        first_token_event.set()
        if len(self.started) == 3:
            self.all_started.set()
        await self.release_final.wait()
        yield FakeRequestOutput([1], prompt)


class GeneratorFirstTokenTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def make_generator(model):
        generator = object.__new__(VLLMGenerator)
        generator.use_persistent_async_engine = True
        generator.use_lora = False
        generator.async_request_counter = iter(range(100))
        generator.model = model
        generator.use_native_first_token_event = bool(
            getattr(model, "supports_mnemorag_first_token_event", False)
        )
        generator.use_native_prefill_start_event = bool(
            getattr(model, "supports_mnemorag_prefill_start_event", False)
        )
        generator._build_sampling_params = lambda _params, _scores: SimpleNamespace(
            output_kind=None
        )
        generator._format_outputs = lambda outputs, *_args: outputs
        return generator

    @staticmethod
    def fake_vllm_modules():
        sampling_module = types.ModuleType("vllm.sampling_params")
        sampling_module.RequestOutputKind = SimpleNamespace(
            CUMULATIVE="cumulative", FINAL_ONLY="final_only"
        )
        vllm_module = types.ModuleType("vllm")
        vllm_module.sampling_params = sampling_module
        return {
            "vllm": vllm_module,
            "vllm.sampling_params": sampling_module,
        }

    async def test_empty_stream_update_does_not_open_first_token_gate(self):
        first_token_event = asyncio.Event()
        model = FakeStreamingModel(first_token_event)
        generator = self.make_generator(model)

        with patch.dict(sys.modules, self.fake_vllm_modules()):
            outputs = await generator.generate_async(
                ["prompt"], first_token_event=first_token_event
            )

        self.assertTrue(model.empty_update_was_ignored)
        self.assertTrue(model.token_update_opened_gate)
        self.assertEqual(outputs[0].outputs[0].token_ids, [7, 8])
        self.assertEqual(outputs[0].outputs[0].text, "first second")
        self.assertEqual(generator.async_request_count, 1)
        self.assertEqual(generator.async_stream_update_count, 3)

    async def test_batch_gate_waits_for_every_request_first_token(self):
        first_token_event = asyncio.Event()
        fast_ready = asyncio.Event()
        release_slow = asyncio.Event()
        generator = self.make_generator(
            StaggeredStreamingModel(fast_ready, release_slow)
        )

        with patch.dict(sys.modules, self.fake_vllm_modules()):
            task = asyncio.create_task(
                generator.generate_async(
                    ["fast", "slow"], first_token_event=first_token_event
                )
            )
            await asyncio.wait_for(fast_ready.wait(), timeout=0.2)
            self.assertFalse(first_token_event.is_set())
            release_slow.set()
            await asyncio.wait_for(task, timeout=0.2)

        self.assertTrue(first_token_event.is_set())

    async def test_native_first_token_event_uses_final_only_output(self):
        first_token_event = asyncio.Event()
        first_token_ready = asyncio.Event()
        release_final = asyncio.Event()
        model = NativeFirstTokenModel(first_token_ready, release_final)
        generator = self.make_generator(model)

        with patch.dict(sys.modules, self.fake_vllm_modules()):
            task = asyncio.create_task(
                generator.generate_async(
                    ["prompt"], first_token_event=first_token_event
                )
            )
            await asyncio.wait_for(first_token_ready.wait(), timeout=0.2)
            await asyncio.wait_for(first_token_event.wait(), timeout=0.2)
            self.assertFalse(task.done())
            self.assertEqual(model.output_kind, "final_only")
            self.assertEqual(first_token_event.mnemorag_set_time_s, 123.0)
            release_final.set()
            outputs = await asyncio.wait_for(task, timeout=0.2)

        self.assertEqual(outputs[0].outputs[0].text, "final")
        self.assertEqual(generator.async_request_count, 1)
        self.assertEqual(generator.async_stream_update_count, 1)

    async def test_prefill_start_gate_uses_first_request_admission(self):
        prefill_start_event = asyncio.Event()
        fast_ready = asyncio.Event()
        release_slow = asyncio.Event()
        generator = self.make_generator(StaggeredPrefillModel(fast_ready, release_slow))

        with patch.dict(sys.modules, self.fake_vllm_modules()):
            task = asyncio.create_task(
                generator.generate_async(
                    ["fast", "slow"],
                    prefill_start_event=prefill_start_event,
                )
            )
            await asyncio.wait_for(fast_ready.wait(), timeout=0.2)
            self.assertTrue(prefill_start_event.is_set())
            self.assertEqual(prefill_start_event.mnemorag_set_time_s, 10.0)
            release_slow.set()
            await asyncio.wait_for(task, timeout=0.2)

        self.assertTrue(prefill_start_event.is_set())
        self.assertEqual(prefill_start_event.mnemorag_set_time_s, 10.0)

    async def test_batch_prefill_gate_waits_for_every_request_admission(self):
        prefill_start_event = asyncio.Event()
        fast_ready = asyncio.Event()
        release_slow = asyncio.Event()
        generator = self.make_generator(StaggeredPrefillModel(fast_ready, release_slow))

        with patch.dict(sys.modules, self.fake_vllm_modules()):
            task = asyncio.create_task(
                generator.generate_async(
                    ["fast", "slow"],
                    prefill_start_event=prefill_start_event,
                    prefill_start_after_all=True,
                )
            )
            await asyncio.wait_for(fast_ready.wait(), timeout=0.2)
            self.assertFalse(prefill_start_event.is_set())
            release_slow.set()
            await asyncio.wait_for(task, timeout=0.2)

        self.assertTrue(prefill_start_event.is_set())
        self.assertEqual(prefill_start_event.mnemorag_set_time_s, 20.0)

    async def test_request_failure_cancels_sibling_generation(self):
        sibling_started = asyncio.Event()
        sibling_cancelled = asyncio.Event()
        generator = self.make_generator(
            PartialFailureModel(sibling_started, sibling_cancelled)
        )

        with patch.dict(sys.modules, self.fake_vllm_modules()):
            with self.assertRaisesRegex(RuntimeError, "injected request failure"):
                await asyncio.wait_for(
                    generator.generate_async(["blocked", "fail"]), timeout=0.2
                )

        self.assertTrue(sibling_cancelled.is_set())

    async def test_async_admission_bounds_sequences_and_releases_permits(self):
        two_started = asyncio.Event()
        release = asyncio.Event()
        model = AdmissionBoundModel(two_started, release)
        generator = self.make_generator(model)
        generator.async_admission_max_num_seqs = 2
        generator.async_admission_max_prefill_tokens = 12
        generator.tokenizer = lambda _prompt, **_kwargs: {
            "input_ids": [1, 2, 3, 4, 5, 6]
        }

        with patch.dict(sys.modules, self.fake_vllm_modules()):
            task = asyncio.create_task(
                generator.generate_async(["one", "two", "three"])
            )
            await asyncio.wait_for(two_started.wait(), timeout=0.2)
            await asyncio.sleep(0)
            self.assertEqual(len(model.started), 2)
            running_stats = generator.get_async_admission_stats()
            self.assertEqual(running_stats["active_sequences"], 2)
            self.assertLessEqual(running_stats["max_prefill_tokens_in_use"], 12)
            release.set()
            outputs = await asyncio.wait_for(task, timeout=0.2)

        self.assertEqual(len(outputs), 3)
        final_stats = generator.get_async_admission_stats()
        self.assertEqual(final_stats["max_active_sequences"], 2)
        self.assertEqual(final_stats["total_admissions"], 3)
        self.assertEqual(final_stats["completed_admissions"], 3)
        self.assertFalse(final_stats["permit_leak"])

    async def test_prefill_token_budget_releases_at_first_generated_token(self):
        first_started = asyncio.Event()
        release_first_token = asyncio.Event()
        all_started = asyncio.Event()
        release_final = asyncio.Event()
        model = TokenBudgetModel(
            first_started,
            release_first_token,
            all_started,
            release_final,
        )
        generator = self.make_generator(model)
        generator.async_admission_max_num_seqs = 3
        generator.async_admission_max_prefill_tokens = 6
        generator.tokenizer = lambda _prompt, **_kwargs: {
            "input_ids": [1, 2, 3, 4, 5, 6]
        }

        with patch.dict(sys.modules, self.fake_vllm_modules()):
            task = asyncio.create_task(
                generator.generate_async(["one", "two", "three"])
            )
            await asyncio.wait_for(first_started.wait(), timeout=0.2)
            await asyncio.sleep(0)
            self.assertEqual(model.started, ["one"])
            blocked_stats = generator.get_async_admission_stats()
            self.assertEqual(blocked_stats["prefill_tokens_in_use"], 6)
            self.assertEqual(blocked_stats["queued_admissions"], 2)

            release_first_token.set()
            await asyncio.wait_for(all_started.wait(), timeout=0.2)
            running_stats = generator.get_async_admission_stats()
            self.assertEqual(running_stats["active_sequences"], 3)
            self.assertLessEqual(running_stats["max_prefill_tokens_in_use"], 6)
            self.assertEqual(
                running_stats["prefill_token_release_boundary"],
                "first_generated_token_or_request_completion",
            )
            release_final.set()
            outputs = await asyncio.wait_for(task, timeout=0.2)

        self.assertEqual(len(outputs), 3)
        self.assertFalse(generator.get_async_admission_stats()["permit_leak"])


if __name__ == "__main__":
    unittest.main()
