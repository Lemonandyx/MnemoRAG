#!/usr/bin/env python3
"""Continuously audit CPU affinity and target-GPU ownership."""

from __future__ import annotations

import os
import statistics
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path


def _parse_cpu_list(value: str) -> set[int]:
    cpu_ids = set()
    for item in value.strip().split(","):
        if not item:
            continue
        if "-" in item:
            start, end = map(int, item.split("-", 1))
            cpu_ids.update(range(start, end + 1))
        else:
            cpu_ids.add(int(item))
    return cpu_ids


def physical_sibling_cpu_ids(cpu_ids: list[int]) -> list[int]:
    """Return every online SMT sibling of the selected physical cores."""
    siblings = set(map(int, cpu_ids))
    for cpu_id in cpu_ids:
        path = Path(
            f"/sys/devices/system/cpu/cpu{cpu_id}/topology/thread_siblings_list"
        )
        try:
            siblings.update(_parse_cpu_list(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(siblings)


def _read_cpu_counters(cpu_ids: list[int]) -> dict[int, tuple[int, int]]:
    selected = set(map(int, cpu_ids))
    counters = {}
    with open("/proc/stat", encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if not fields or not fields[0].startswith("cpu"):
                continue
            suffix = fields[0][3:]
            if not suffix.isdigit() or int(suffix) not in selected:
                continue
            values = [int(value) for value in fields[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            counters[int(suffix)] = (sum(values), idle)
    return counters


def _read_parent_pid(pid: int) -> int | None:
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    parent = next(
        (line for line in status.splitlines() if line.startswith("PPid:")),
        None,
    )
    return int(parent.split()[1]) if parent is not None else None


def _process_tree_pids(root_pid: int) -> set[int]:
    children: dict[int, list[int]] = {}
    for path in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(path.name)
        except ValueError:
            continue
        parent = _read_parent_pid(pid)
        if parent is not None:
            children.setdefault(parent, []).append(pid)
    observed = {int(root_pid)}
    pending = [int(root_pid)]
    while pending:
        parent = pending.pop()
        for child in children.get(parent, []):
            if child not in observed:
                observed.add(child)
                pending.append(child)
    return observed


def _task_ticks(path: Path) -> int | None:
    try:
        line = path.read_text(encoding="utf-8")
        fields = line[line.rfind(")") + 2 :].split()
        return int(fields[11]) + int(fields[12])
    except (OSError, ValueError, IndexError):
        return None


def _read_process_tree_ticks(pids: set[int]) -> dict[int, int]:
    ticks = {}
    for pid in pids:
        total = 0
        observed = False
        for path in Path(f"/proc/{pid}/task").glob("[0-9]*/stat"):
            value = _task_ticks(path)
            if value is not None:
                total += value
                observed = True
        if observed:
            ticks[pid] = total
    return ticks


def _thread_affinity_report(
    pids: set[int], expected_cpu_ids: list[int], root_pid: int | None = None
) -> dict:
    expected = set(map(int, expected_cpu_ids))
    outside = set()
    masks = Counter()
    thread_count = 0
    process_count = 0
    root_affinity_matches = False
    root_pid = int(root_pid if root_pid is not None else os.getpid())
    for pid in sorted(pids):
        observed_process = False
        for path in Path(f"/proc/{pid}/task").glob("[0-9]*"):
            try:
                task_id = int(path.name)
                allowed = set(os.sched_getaffinity(task_id))
            except (ValueError, FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            observed_process = True
            thread_count += 1
            masks[tuple(sorted(allowed))] += 1
            outside.update(allowed - expected)
            if pid == root_pid and task_id == pid:
                root_affinity_matches = allowed == expected
        process_count += int(observed_process)
    return {
        "process_count": process_count,
        "thread_count": thread_count,
        "root_affinity_matches_expected": root_affinity_matches,
        "outside_cpu_ids": sorted(outside),
        "affinity_mask_count": len(masks),
    }


def _belongs_to_tree(pid: int, root_pid: int) -> bool:
    current = int(pid)
    seen = set()
    while current > 1 and current not in seen:
        if current == root_pid:
            return True
        seen.add(current)
        parent = _read_parent_pid(current)
        if parent is None:
            return False
        current = parent
    return False


def _gpu_processes(gpu_id: str, root_pid: int) -> list[dict]:
    command = [
        "nvidia-smi",
        f"--id={gpu_id}",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    output = None
    for timeout_s in (5, 10):
        try:
            output = subprocess.check_output(
                command,
                text=True,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            ).strip()
            break
        except subprocess.TimeoutExpired:
            if timeout_s == 10:
                raise
    if output is None:
        raise RuntimeError("GPU process audit produced no output")
    processes = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        pid = int(fields[0])
        try:
            memory = float(fields[1])
        except ValueError:
            memory = None
        processes.append(
            {
                "pid": pid,
                "used_memory_mib": memory,
                "owned_by_benchmark": _belongs_to_tree(pid, root_pid),
            }
        )
    return processes


def _external_cpu_busy_percent(
    previous_counters: dict[int, tuple[int, int]] | None,
    counters: dict[int, tuple[int, int]],
    previous_ticks: dict[int, int] | None,
    ticks: dict[int, int],
    capacity_cpu_ids: list[int] | None = None,
) -> float | None:
    if previous_counters is None or previous_ticks is None:
        return None
    if set(previous_counters) != set(counters):
        return None
    capacity = set(capacity_cpu_ids or counters)
    total_delta = 0
    busy_delta = 0
    for cpu_id, (total, idle) in counters.items():
        old_total, old_idle = previous_counters[cpu_id]
        cpu_total = max(0, total - old_total)
        cpu_idle = max(0, idle - old_idle)
        if cpu_id in capacity:
            total_delta += cpu_total
        busy_delta += max(0, cpu_total - cpu_idle)
    if total_delta <= 0:
        return None
    own_delta = sum(
        max(0, value - previous_ticks.get(pid, 0)) for pid, value in ticks.items()
    )
    return 100.0 * max(0, busy_delta - own_delta) / total_delta


class RuntimeResourceMonitor:
    def __init__(
        self,
        selected_cpu_ids: list[int],
        gpu_id: str,
        *,
        require_clean: bool,
        max_external_cpu_busy_percent: float,
        sample_interval_s: float,
        monitored_cpu_ids: list[int] | None = None,
    ) -> None:
        if sample_interval_s <= 0:
            raise ValueError("runtime resource sample interval must be positive")
        self.selected_cpu_ids = sorted(map(int, selected_cpu_ids))
        self.monitored_cpu_ids = sorted(
            map(
                int,
                monitored_cpu_ids
                if monitored_cpu_ids is not None
                else physical_sibling_cpu_ids(self.selected_cpu_ids),
            )
        )
        self.gpu_id = str(gpu_id)
        self.require_clean = bool(require_clean)
        self.max_external_cpu_busy_percent = float(max_external_cpu_busy_percent)
        self.sample_interval_s = float(sample_interval_s)
        self.root_pid = os.getpid()
        self.samples: list[dict] = []
        self.violations: list[str] = []
        self._previous_counters = None
        self._previous_ticks = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sample_index = 0
        self._monitor_errors: list[str] = []

    def sample_once(self, label: str | None = None) -> dict:
        with self._lock:
            pids = _process_tree_pids(self.root_pid)
            counters = _read_cpu_counters(self.monitored_cpu_ids)
            ticks = _read_process_tree_ticks(pids)
            external_cpu = _external_cpu_busy_percent(
                self._previous_counters,
                counters,
                self._previous_ticks,
                ticks,
                self.selected_cpu_ids,
            )
            self._previous_counters = counters
            self._previous_ticks = ticks
            affinity = _thread_affinity_report(
                pids, self.selected_cpu_ids, self.root_pid
            )
            gpu_error = None
            try:
                gpu_processes = _gpu_processes(self.gpu_id, self.root_pid)
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                gpu_processes = []
                gpu_error = f"{type(error).__name__}: {error}"
            foreign_gpu = [
                process
                for process in gpu_processes
                if not process["owned_by_benchmark"]
            ]
            self._sample_index += 1
            sample = {
                "sample": self._sample_index,
                "label": label or f"periodic:{self._sample_index}",
                "monotonic_s": time.perf_counter(),
                "system_load_1m": float(os.getloadavg()[0]),
                "selected_cpu_count": len(self.selected_cpu_ids),
                "external_cpu_busy_percent": external_cpu,
                "process_tree_pid_count": len(pids),
                "thread_affinity": affinity,
                "gpu_compute_processes": gpu_processes,
                "foreign_gpu_compute_processes": foreign_gpu,
                "gpu_monitor_error": gpu_error,
            }
            sample_violations = []
            if set(counters) != set(self.monitored_cpu_ids):
                sample_violations.append(
                    "selected physical-core sibling counters are incomplete"
                )
            if not affinity["root_affinity_matches_expected"]:
                sample_violations.append("root process CPU affinity changed")
            if affinity["outside_cpu_ids"]:
                sample_violations.append(
                    "process-tree thread affinity escaped selected CPUs: "
                    f"{affinity['outside_cpu_ids']}"
                )
            if (
                external_cpu is not None
                and external_cpu > self.max_external_cpu_busy_percent
            ):
                sample_violations.append(
                    f"external CPU busy {external_cpu:.2f}% exceeds "
                    f"{self.max_external_cpu_busy_percent:.2f}%"
                )
            if gpu_error is not None:
                sample_violations.append(f"GPU monitor failed: {gpu_error}")
            if foreign_gpu:
                sample_violations.append(
                    f"foreign GPU compute processes: {foreign_gpu}"
                )
            sample["violations"] = sample_violations
            self.samples.append(sample)
            self.violations.extend(
                f"{sample['label']}: {violation}" for violation in sample_violations
            )
            return sample

    def _run(self) -> None:
        while not self._stop.wait(self.sample_interval_s):
            try:
                self.sample_once()
            except BaseException as error:
                message = f"{type(error).__name__}: {error}"
                with self._lock:
                    self._monitor_errors.append(message)
                    self.violations.append(f"continuous monitor failed: {message}")
                self._stop.set()
                return

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("runtime resource monitor already started")
        self.sample_once("monitor:start")
        self._thread = threading.Thread(
            target=self._run,
            name="mnemorag-resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def checkpoint(self, label: str) -> dict:
        return self.sample_once(label)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, 2.0 * self.sample_interval_s))
            self._thread = None

    def assert_clean(self) -> None:
        if self.require_clean and self.violations:
            raise RuntimeError(
                "Runtime resource contamination: "
                + "; ".join(dict.fromkeys(self.violations))
            )

    def summary(self) -> dict:
        external = [
            sample["external_cpu_busy_percent"]
            for sample in self.samples
            if sample["external_cpu_busy_percent"] is not None
        ]
        return {
            "status": "clean" if not self.violations else "contaminated",
            "continuous": True,
            "sample_interval_s": self.sample_interval_s,
            "sample_count": len(self.samples),
            "selected_cpu_ids": self.selected_cpu_ids,
            "monitored_sibling_cpu_ids": self.monitored_cpu_ids,
            "external_cpu_busy_percent_mean": (
                statistics.fmean(external) if external else None
            ),
            "external_cpu_busy_percent_max": max(external) if external else None,
            "foreign_gpu_process_count_max": max(
                (
                    len(sample["foreign_gpu_compute_processes"])
                    for sample in self.samples
                ),
                default=0,
            ),
            "monitor_errors": list(self._monitor_errors),
            "violations": list(dict.fromkeys(self.violations)),
            "samples": self.samples,
        }
