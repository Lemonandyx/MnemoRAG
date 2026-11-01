import math
from types import SimpleNamespace

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock


def _queue(monkeypatch, half_life_s=10.0, size=3):
    monkeypatch.setenv("VLLM_KV_FREE_BLOCK_POLICY", "heat")
    monkeypatch.setenv("VLLM_KV_HEAT_HALF_LIFE_S", str(half_life_s))
    monkeypatch.delenv("VLLM_KV_HEAT_FREQUENCY_LRU", raising=False)
    monkeypatch.delenv("VLLM_KV_HEAT_PURE_LFU", raising=False)
    monkeypatch.delenv("VLLM_KV_PROFILE_NEXT_USE_PATH", raising=False)
    monkeypatch.delenv("VLLM_KV_ORACLE_NEXT_USE_PATH", raising=False)
    return FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(size)])


def _replace(queue, block, *, heat, last_used_ts):
    block.heat = heat
    block.last_used_ts = last_used_ts
    queue.append(block)


def test_heat_policy_prefers_unused_blocks(monkeypatch):
    queue = _queue(monkeypatch, size=2)
    first = queue.popleft()
    first.heat = 1
    first.last_used_ts = 1.0
    queue.append(first)

    assert queue.popleft().heat == 0


def test_heat_policy_uses_lru_as_equal_frequency_tiebreaker(monkeypatch):
    queue = _queue(monkeypatch, size=2)
    older = queue.popleft()
    newer = queue.popleft()
    _replace(queue, older, heat=4, last_used_ts=10.0)
    _replace(queue, newer, heat=4, last_used_ts=20.0)

    assert queue.popleft() is older


def test_heat_policy_retains_hotter_block_at_equal_recency(monkeypatch):
    queue = _queue(monkeypatch, size=2)
    colder = queue.popleft()
    hotter = queue.popleft()
    _replace(queue, colder, heat=1, last_used_ts=10.0)
    _replace(queue, hotter, heat=4, last_used_ts=10.0)

    assert queue.popleft() is colder


def test_heat_policy_half_life_balances_frequency_and_recency(monkeypatch):
    queue = _queue(monkeypatch, half_life_s=10.0, size=2)
    hot_old = queue.popleft()
    cold_recent = queue.popleft()
    _replace(queue, hot_old, heat=4, last_used_ts=0.0)
    _replace(queue, cold_recent, heat=1, last_used_ts=10.0)

    expected_hot = math.log(4)
    expected_cold = math.log(1) + math.log(2)
    assert queue.calc_priority(hot_old) == pytest.approx(expected_hot)
    assert queue.calc_priority(cold_recent) == pytest.approx(expected_cold)
    assert queue.popleft() is cold_recent


def test_heat_policy_rejects_nonpositive_half_life(monkeypatch):
    monkeypatch.setenv("VLLM_KV_FREE_BLOCK_POLICY", "heat")
    monkeypatch.setenv("VLLM_KV_HEAT_HALF_LIFE_S", "0")

    with pytest.raises(ValueError, match="must be positive"):
        FreeKVCacheBlockQueue([KVCacheBlock(0)])


def test_heat_heap_compaction_bounds_stale_entries(monkeypatch):
    monkeypatch.setenv("VLLM_KV_HEAP_COMPACTION_FACTOR", "2")
    monkeypatch.setenv("VLLM_KV_HEAP_COMPACTION_MIN_STALE", "0")
    queue = _queue(monkeypatch, size=2)
    first = queue.popleft()
    second = queue.popleft()
    _replace(queue, first, heat=1, last_used_ts=1.0)
    _replace(queue, second, heat=1, last_used_ts=1.0)

    for access in range(2, 100):
        queue.remove(first)
        first.heat = access
        first.last_used_ts = float(access)
        queue.append(first)

    assert queue.heap_compactions > 0
    assert len(queue.heap) <= 2 * queue.num_free_blocks
    assert queue.popleft() is second


def test_pattern_heat_metadata_is_capacity_bounded(monkeypatch):
    monkeypatch.setenv("VLLM_KV_FREE_BLOCK_POLICY", "heat")
    monkeypatch.setenv("VLLM_KV_PATTERN_METADATA_CAPACITY", "16")
    pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=True,
        hash_block_size=16,
    )
    block = pool.blocks[1]

    for block_hash in range(100):
        pool._record_block_access(
            block_hash,
            block,
            block_index=0,
            prompt_full_blocks=1,
            num_cached_blocks=0,
            cache_hit=False,
            request_position=block_hash,
        )

    stats = pool.get_mnemorag_stats()
    assert stats["pattern_heat_entries"] == 16
    assert stats["pattern_metadata_evictions"]["heat"] == 84
    assert 0 not in pool.block_pattern_heat
    assert 99 in pool.block_pattern_heat


@pytest.mark.parametrize("policy", ["lru", "fifo"])
def test_non_heat_policy_skips_pattern_heat_metadata(monkeypatch, policy):
    monkeypatch.setenv("VLLM_KV_FREE_BLOCK_POLICY", policy)
    monkeypatch.setenv("VLLM_KV_PATTERN_METADATA_CAPACITY", "16")
    pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=True,
        hash_block_size=16,
    )
    block = pool.blocks[1]

    for block_hash in range(100):
        pool._record_block_access(
            block_hash,
            block,
            block_index=0,
            prompt_full_blocks=1,
            num_cached_blocks=0,
            cache_hit=False,
            request_position=block_hash,
        )

    assert pool.get_mnemorag_stats()["pattern_heat_entries"] == 0


def test_document_heat_is_weighted_by_block_token_coverage(monkeypatch):
    monkeypatch.setenv("VLLM_KV_FREE_BLOCK_POLICY", "heat")
    monkeypatch.setenv("VLLM_KV_HEAT_SOURCE", "document")
    pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=True,
        hash_block_size=16,
    )
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(
            extra_args={
                "mnemorag_document_heat_spans": [
                    [4, 12, 10.0],
                    [12, 20, 2.0],
                ]
            }
        )
    )

    assert pool._get_document_block_heat(request, 0, 16) == pytest.approx(
        (8 * 10.0 + 4 * 2.0) / 12
    )
    assert pool._get_document_block_heat(request, 1, 16) == pytest.approx(2.0)
    stats = pool.get_mnemorag_stats()
    assert stats["document_heat_accesses"] == 2
    assert stats["document_heat_blocks"] == 2


def test_document_heat_replaces_prefix_frequency_for_document_blocks(monkeypatch):
    monkeypatch.setenv("VLLM_KV_FREE_BLOCK_POLICY", "heat")
    monkeypatch.setenv("VLLM_KV_HEAT_SOURCE", "document")
    pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=True,
        hash_block_size=16,
    )
    block = pool.blocks[1]

    pool._record_block_access(
        b"document-block",
        block,
        block_index=1,
        prompt_full_blocks=3,
        num_cached_blocks=0,
        cache_hit=False,
        request_position=1,
        document_heat=7.0,
    )
    assert block.heat == pytest.approx(7.0)

    pool._record_block_access(
        b"document-block",
        block,
        block_index=1,
        prompt_full_blocks=3,
        num_cached_blocks=1,
        cache_hit=True,
        request_position=2,
        document_heat=12.0,
    )
    assert block.heat == pytest.approx(12.0)
