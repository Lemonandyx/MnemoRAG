#!/usr/bin/env python3
"""Build a reproducible document-frequency profile from a retrieval trace."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("requests_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--minimum-frequency", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.minimum_frequency < 1:
        raise ValueError("--minimum-frequency must be at least 1")

    counts: Counter[int | str] = Counter()
    request_count = 0
    with args.requests_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if "retrieval_idxs" not in (reader.fieldnames or []):
            raise ValueError("requests CSV is missing retrieval_idxs")
        for row in reader:
            ids = ast.literal_eval(row["retrieval_idxs"])
            if not isinstance(ids, (list, tuple)):
                raise ValueError("retrieval_idxs must decode to a list or tuple")
            counts.update(ids)
            request_count += 1

    prior = {
        str(doc_id): int(count)
        for doc_id, count in counts.items()
        if count >= args.minimum_frequency
    }
    write_json(args.output_json, prior)
    output_sha256 = file_sha256(args.output_json)

    manifest_path = args.manifest or args.output_json.with_suffix(
        args.output_json.suffix + ".manifest.json"
    )
    manifest = {
        "status": "complete",
        "profile_kind": "paired_trace_document_frequency",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_requests_csv": str(args.requests_csv.resolve()),
        "source_requests_csv_sha256": file_sha256(args.requests_csv),
        "request_count": request_count,
        "retrieved_document_observations": int(sum(counts.values())),
        "unique_documents": len(counts),
        "minimum_frequency": args.minimum_frequency,
        "profiled_documents": len(prior),
        "profiled_document_observations": int(sum(prior.values())),
        "output_json": str(args.output_json.resolve()),
        "output_json_sha256": output_sha256,
        "causality_note": (
            "This profile is derived from the paired evaluation retrieval trace "
            "and represents profile-guided, not purely online, Heat."
        ),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
