import asyncio

from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreOutput,
)
from vllm.v1.engine.output_processor import OutputProcessor


def test_prefill_event_uses_engine_scheduler_admission_timestamp():
    async def run_test():
        processor = object.__new__(OutputProcessor)
        event = asyncio.Event()
        processor.mnemorag_prefill_start_events = {"request-0": event}
        output = EngineCoreOutput(
            request_id="request-0",
            new_token_ids=[],
            events=[
                EngineCoreEvent.new_event(
                    EngineCoreEventType.QUEUED, timestamp=10.0
                ),
                EngineCoreEvent.new_event(
                    EngineCoreEventType.SCHEDULED, timestamp=12.5
                ),
            ],
        )

        processor._signal_mnemorag_prefill_start_event(output)

        assert event.is_set()
        assert event.mnemorag_set_time_s == 12.5
        assert processor.mnemorag_prefill_start_events == {}

    asyncio.run(run_test())


def test_prefill_event_waits_until_scheduled_event_exists():
    async def run_test():
        processor = object.__new__(OutputProcessor)
        event = asyncio.Event()
        processor.mnemorag_prefill_start_events = {"request-0": event}
        output = EngineCoreOutput(
            request_id="request-0",
            new_token_ids=[],
            events=[
                EngineCoreEvent.new_event(
                    EngineCoreEventType.QUEUED, timestamp=10.0
                )
            ],
        )

        processor._signal_mnemorag_prefill_start_event(output)

        assert not event.is_set()
        assert "request-0" in processor.mnemorag_prefill_start_events

    asyncio.run(run_test())
