import json

from benchmarks.build_next_use_profile import build_profiles


def test_build_profiles_excludes_one_off_cpu_blocks(tmp_path):
    access_log = tmp_path / "access.jsonl"
    access_log.write_text(
        "\n".join(
            json.dumps({"block_hash": block_hash, "request_position": position})
            for block_hash, position in (("aa", 0), ("bb", 0), ("aa", 5), ("aa", 5))
        )
        + "\n",
        encoding="utf-8",
    )
    gpu_output = tmp_path / "gpu.json"
    cpu_output = tmp_path / "cpu.json"
    manifest_output = tmp_path / "manifest.json"

    manifest = build_profiles(
        access_log, gpu_output, cpu_output, manifest_output
    )

    assert json.loads(gpu_output.read_text()) == {"aa": [0, 5], "bb": [0]}
    assert json.loads(cpu_output.read_text()) == ["aa"]
    assert manifest["recorded_lines"] == 4
    assert manifest["unique_block_request_pairs"] == 3
    assert manifest["productive_cpu_hashes"] == 1
