#!/usr/bin/env python3
"""Build GPU next-use and CPU productive-admission profiles from one trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_profiles(
    access_log: Path,
    gpu_output: Path,
    cpu_output: Path,
    manifest_output: Path,
) -> dict[str, object]:
    accesses: dict[str, set[int]] = {}
    recorded_lines = 0
    with access_log.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            block_hash = str(record["block_hash"])
            try:
                raw_hash = bytes.fromhex(block_hash)
            except ValueError as error:
                raise ValueError(
                    f"invalid block hash on line {line_number}"
                ) from error
            if not raw_hash:
                raise ValueError(f"empty block hash on line {line_number}")
            request_position = int(record["request_position"])
            if request_position < 0:
                raise ValueError(
                    f"negative request position on line {line_number}"
                )
            accesses.setdefault(block_hash, set()).add(request_position)
            recorded_lines += 1

    gpu_profile = {
        block_hash: sorted(positions)
        for block_hash, positions in sorted(accesses.items())
    }
    # CPU admission excludes one-off blocks. These blocks cannot produce a
    # future H2D hit and previously consumed most of the nominal offload tier.
    cpu_profile = sorted(
        block_hash for block_hash, positions in gpu_profile.items()
        if len(positions) > 1
    )
    atomic_write_json(gpu_output, gpu_profile)
    atomic_write_json(cpu_output, cpu_profile)

    manifest = {
        "format": "mnemorag.full-nq-next-use-profile.v1",
        "status": "complete",
        "semantics": {
            "gpu": "offline_next_use_positions",
            "cpu": "productive_admission_hashes_with_at_least_two_requests",
        },
        "access_log": str(access_log.resolve()),
        "access_log_sha256": file_sha256(access_log),
        "gpu_profile": str(gpu_output.resolve()),
        "gpu_profile_sha256": file_sha256(gpu_output),
        "cpu_profile": str(cpu_output.resolve()),
        "cpu_profile_sha256": file_sha256(cpu_output),
        "recorded_lines": recorded_lines,
        "unique_block_request_pairs": sum(map(len, gpu_profile.values())),
        "unique_block_hashes": len(gpu_profile),
        "productive_cpu_hashes": len(cpu_profile),
        "maximum_request_position": max(
            (position for positions in gpu_profile.values() for position in positions),
            default=None,
        ),
    }
    atomic_write_json(manifest_output, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--access-log", type=Path, required=True)
    parser.add_argument("--gpu-output", type=Path, required=True)
    parser.add_argument("--cpu-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_profiles(
        args.access_log,
        args.gpu_output,
        args.cpu_output,
        args.manifest_output,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
