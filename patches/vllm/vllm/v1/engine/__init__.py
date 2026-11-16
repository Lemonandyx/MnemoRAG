# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections.abc import Mapping
from typing import Any

import msgspec
import torch

from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import LogprobsLists, LogprobsTensors
from vllm.v1.serial_utils import UtilityResult

# Completion reasons form part of the request-output API.
FINISH_REASON_STRINGS = ("stop", "length", "abort")


class FinishReason(enum.IntEnum):
    """Request completion reason: stop, length limit, or abort."""

    STOP = 0
    LENGTH = 1
    ABORT = 2

    def __str__(self):
        return FINISH_REASON_STRINGS[self.value]


class EngineCoreRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec] | None
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    eos_token_id: int | None
    arrival_time: float
    lora_request: LoRARequest | None
    cache_salt: str | None
    data_parallel_rank: int | None
    prompt_embeds: torch.Tensor | None = None

    # Route outputs back to the originating client.
    client_index: int = 0

    # Track the data-parallel wave when requests race with completion notifications.
    current_wave: int = 0
    priority: int = 0

    trace_headers: Mapping[str, str] | None = None

    @property
    def params(self) -> SamplingParams | PoolingParams:
        """Return the processed params (sampling or pooling)."""
        if self.sampling_params is not None:
            return self.sampling_params
        assert self.pooling_params is not None
        return self.pooling_params


class EngineCoreEventType(enum.IntEnum):
    """The type of engine core request event."""

    QUEUED = 1
    SCHEDULED = 2
    PREEMPTED = 3


class EngineCoreEvent(msgspec.Struct):
    """Engine event timestamped by a monotonic clock."""

    type: EngineCoreEventType
    timestamp: float

    @classmethod
    def new_event(
        cls, event_type: EngineCoreEventType, timestamp: float | None = None
    ) -> "EngineCoreEvent":
        timestamp = time.monotonic() if timestamp is None else timestamp
        return cls(event_type, timestamp)


class EngineCoreOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    request_id: str
    new_token_ids: list[int]

    new_logprobs: LogprobsLists | None = None
    new_prompt_logprobs_tensors: LogprobsTensors | None = None

    pooling_output: torch.Tensor | None = None

    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    events: list[EngineCoreEvent] | None = None
    kv_transfer_params: dict[str, Any] | None = None

    trace_headers: Mapping[str, str] | None = None
    # The number of tokens with prefix cache hits.
    num_cached_tokens: int = 0

    # The number of NaNs in logits.
    # A value greater than 0 indicates that the output is corrupted.
    num_nans_in_logits: int = 0

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


class UtilityOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    call_id: int

    # Non-None implies the call failed, result should be None.
    failure_message: str | None = None
    result: UtilityResult | None = None


class EngineCoreOutputs(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]

    engine_index: int = 0

    # [num_reqs]
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: SchedulerStats | None = None
    timestamp: float = 0.0

    utility_output: UtilityOutput | None = None
    finished_requests: set[str] | None = None

    # In DP case, used to signal that the current wave of requests
    # has finished and the engines are paused.
    wave_complete: int | None = None
    # In DP case, used to signal that a request was received for an
    # "old" wave, so the next wave needs to be started in other engines.
    start_wave: int | None = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


class EngineCoreRequestType(enum.Enum):
    """Socket request types encoded as byte strings."""

    ADD = b"\x00"
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    UTILITY = b"\x03"
    # Sentinel used within EngineCoreProc.
    EXECUTOR_FAILED = b"\x04"


class ReconfigureDistributedRequest(msgspec.Struct):
    new_data_parallel_size: int
    new_data_parallel_rank: int
    new_data_parallel_rank_local: int
    new_data_parallel_master_ip: str
    new_data_parallel_master_port: int


class ReconfigureRankType(enum.IntEnum):
    """Rank role in a distributed reconfiguration."""

    KEEP_CURRENT_RANK = -1
    SHUTDOWN_CURRENT_RANK = -2
