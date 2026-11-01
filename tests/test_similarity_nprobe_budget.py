import pytest

from benchmarks.benchmark_point import (
    resolve_similarity_nprobe_bounds,
)


def test_similarity_nprobe_bounds_scale_with_nprobe():
    assert resolve_similarity_nprobe_bounds(128, 0.75, 1.0) == (96, 128)
    assert resolve_similarity_nprobe_bounds(256, 0.75, 1.0) == (192, 256)
    assert resolve_similarity_nprobe_bounds(512, 1.0, 1.0) == (512, 512)


def test_similarity_nprobe_bounds_round_up_and_preserve_defaults():
    assert resolve_similarity_nprobe_bounds(128, 0.501, 0.751) == (65, 97)
    assert resolve_similarity_nprobe_bounds(128, None, None) == (None, None)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.01])
def test_similarity_nprobe_bounds_reject_invalid_fractions(value):
    with pytest.raises(ValueError, match=r"must be in \(0, 1\]"):
        resolve_similarity_nprobe_bounds(128, value, 1.0)


def test_similarity_nprobe_bounds_reject_inverted_range():
    with pytest.raises(ValueError, match="minimum must not exceed maximum"):
        resolve_similarity_nprobe_bounds(128, 1.0, 0.75)
