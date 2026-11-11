# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections import defaultdict

from vllm.logger import init_logger
from vllm.v1.kv_offload.abstract import LoadStoreSpec

TransferSpec = tuple[LoadStoreSpec, LoadStoreSpec]
TransferType = tuple[str, str]
TransferResult = tuple[int, bool]

logger = init_logger(__name__)


class OffloadingHandler(ABC):
    """Manage asynchronous KV data transfers for one medium pair."""

    @abstractmethod
    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        pass

    @abstractmethod
    def get_finished(self) -> list[TransferResult]:
        pass

    def wait_for_jobs(self, job_ids: set[int]) -> None:
        """Wait for selected jobs when the handler supports directed waits."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support directed transfer waits"
        )

    def pop_completed_metrics(self, job_ids: set[int]) -> dict[int, dict[str, float]]:
        return {}


class OffloadingWorker:
    """Route transfer jobs to the handler for their source/destination pair."""

    def __init__(self):
        self.handlers: set[OffloadingHandler] = set()
        self.transfer_type_to_handler: dict[TransferType, OffloadingHandler] = {}
        self.job_to_handler: dict[int, OffloadingHandler] = {}
        self.completed_job_metrics: dict[int, dict[str, float]] = {}

    def register_handler(
        self,
        src_cls: type[LoadStoreSpec],
        dst_cls: type[LoadStoreSpec],
        handler: OffloadingHandler,
    ) -> None:
        transfer_type = (src_cls.medium(), dst_cls.medium())
        assert transfer_type not in self.transfer_type_to_handler
        self.handlers.add(handler)
        self.transfer_type_to_handler[transfer_type] = handler

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src, dst = spec
        transfer_type = (src.medium(), dst.medium())
        handler = self.transfer_type_to_handler.get(transfer_type)
        assert handler is not None

        try:
            success = handler.transfer_async(job_id, spec)
        except Exception as error:
            logger.warning(
                "Exception in %r transfer %d: %r",
                transfer_type,
                job_id,
                error,
                exc_info=True,
            )
            return False

        if not success:
            logger.warning("Failed to submit %r transfer %d", transfer_type, job_id)
        else:
            self.job_to_handler[job_id] = handler
            logger.debug("Submitted %r transfer %d: %r", transfer_type, job_id, spec)
        return success

    def wait_for_jobs(self, job_ids: set[int]) -> None:
        jobs_by_handler: dict[OffloadingHandler, set[int]] = defaultdict(set)
        for job_id in job_ids:
            handler = self.job_to_handler.get(job_id)
            if handler is not None:
                jobs_by_handler[handler].add(job_id)
        for handler, handler_job_ids in jobs_by_handler.items():
            handler.wait_for_jobs(handler_job_ids)

    def get_finished(self) -> list[TransferResult]:
        finished: list[TransferResult] = []
        for handler in self.handlers:
            handler_finished = handler.get_finished()
            finished.extend(handler_finished)
            self.completed_job_metrics.update(
                handler.pop_completed_metrics(
                    {job_id for job_id, _ in handler_finished}
                )
            )
        for job_id, _ in finished:
            self.job_to_handler.pop(job_id, None)
        return finished

    def pop_completed_metrics(
        self, job_ids: set[int]
    ) -> dict[int, dict[str, float]]:
        return {
            job_id: self.completed_job_metrics.pop(job_id)
            for job_id in job_ids
            if job_id in self.completed_job_metrics
        }
