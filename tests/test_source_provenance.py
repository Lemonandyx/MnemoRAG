from benchmarks.source_provenance import (
    canonical_sha256,
    capture_source_provenance,
    repository_manifest,
)


def test_repository_manifest_hashes_source_but_ignores_results(tmp_path):
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not runtime source\n", encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()
    (results / "generated.json").write_text("{}\n", encoding="utf-8")

    first = repository_manifest(tmp_path)
    (results / "generated.json").write_text('{"changed": true}\n', encoding="utf-8")
    second = repository_manifest(tmp_path)

    assert first["source_file_count"] == 1
    assert first["source_sha256"] == second["source_sha256"]
    assert canonical_sha256(first["files"]) == first["source_sha256"]


def test_repository_manifest_detects_runtime_source_change(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first = repository_manifest(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = repository_manifest(tmp_path)

    assert first["source_sha256"] != second["source_sha256"]


def test_capture_source_provenance_includes_patched_vllm(tmp_path):
    project = tmp_path / "project"
    hedra = tmp_path / "hedra"
    vllm = tmp_path / "vllm"
    for root in (project, hedra, vllm):
        root.mkdir()
        (root / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")

    provenance = capture_source_provenance(project, hedra, vllm)

    assert set(provenance["repositories"]) == {
        "mnemorag_runtime",
        "hedrarag",
        "patched_vllm",
    }
