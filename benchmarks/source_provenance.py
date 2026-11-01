"""Capture immutable source provenance for long-running formal campaigns."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable

SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".json",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}
IGNORED_PARTS = {".git", "__pycache__", "results", "output", "outputs"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_value(root: Path, arguments: Iterable[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def source_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SOURCE_SUFFIXES
        and not any(part in IGNORED_PARTS for part in path.relative_to(root).parts)
    )


def repository_manifest(root: Path) -> dict:
    root = root.resolve()
    files = [
        {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in source_files(root)
    ]
    source_digest = canonical_sha256(files)
    status = git_value(root, ("status", "--short", "--untracked-files=all"))
    return {
        "root": str(root),
        "git_commit": git_value(root, ("rev-parse", "HEAD")),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "git_status_lines": len(status.splitlines()) if status else 0,
        "source_file_count": len(files),
        "source_sha256": source_digest,
        "files": files,
    }


def capture_source_provenance(
    project_root: Path,
    hedra_root: Path,
    vllm_root: Path,
) -> dict:
    repositories = {
        "mnemorag_runtime": repository_manifest(project_root),
        "hedrarag": repository_manifest(hedra_root),
        "patched_vllm": repository_manifest(vllm_root),
    }
    return {
        "format": "mnemorag.source-provenance.v1",
        "repositories": repositories,
        "combined_source_sha256": canonical_sha256(
            {name: manifest["source_sha256"] for name, manifest in repositories.items()}
        ),
    }
