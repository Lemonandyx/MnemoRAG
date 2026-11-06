import json

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.backend import Backend, BlockStatus
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager


class FakeBackend(Backend):
    def __init__(self, capacity):
        super().__init__(block_size=16, medium="CPU")
        self.capacity = capacity
        self.allocated = set()

    def get_num_free_blocks(self):
        return self.capacity - len(self.allocated)

    def allocate_blocks(self, block_hashes):
        blocks = [BlockStatus() for _ in block_hashes]
        self.allocated.update(map(id, blocks))
        return blocks

    def free(self, block):
        self.allocated.remove(id(block))

    def get_load_store_spec(self, block_hashes, blocks):
        return tuple(block_hashes), tuple(blocks)


def block_hash(value):
    return BlockHash(bytes([value]) * 32)


def test_productive_profile_rejects_one_off_cpu_admission(tmp_path, monkeypatch):
    productive = block_hash(1)
    one_off = block_hash(2)
    profile = tmp_path / "productive.json"
    profile.write_text(json.dumps([bytes(productive).hex()]))
    monkeypatch.setenv("VLLM_CPU_KV_ADMISSION_PROFILE_PATH", str(profile))
    monkeypatch.delenv("VLLM_CPU_KV_PROFILE_MODE", raising=False)
    manager = LRUOffloadingManager(FakeBackend(capacity=2))

    prepared = manager.prepare_store([productive, one_off])

    assert prepared is not None
    assert prepared.block_hashes_to_store == [productive]
    assert manager.rejected_blocks == 1


def test_next_use_profile_admits_only_finite_future_reuse(tmp_path, monkeypatch):
    repeated = block_hash(3)
    one_off = block_hash(4)
    profile = tmp_path / "next-use.json"
    profile.write_text(
        json.dumps(
            {
                bytes(repeated).hex(): [0, 5],
                bytes(one_off).hex(): [0],
            }
        )
    )
    monkeypatch.setenv("VLLM_CPU_KV_ADMISSION_PROFILE_PATH", str(profile))
    monkeypatch.setenv("VLLM_CPU_KV_PROFILE_MODE", "near")
    manager = LRUOffloadingManager(FakeBackend(capacity=2))
    manager.lookup([repeated, one_off])

    prepared = manager.prepare_store([repeated, one_off])

    assert prepared is not None
    assert prepared.block_hashes_to_store == [repeated]
