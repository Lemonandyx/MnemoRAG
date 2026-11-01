import unittest

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.backend import Backend, BlockStatus
from vllm.v1.kv_offload.heat_manager import HeatOffloadingManager


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


class HeatOffloadingManagerTest(unittest.TestCase):
    def test_hot_block_survives_capacity_eviction(self):
        manager = HeatOffloadingManager(FakeBackend(capacity=2))
        hot, cold, incoming = map(block_hash, (1, 2, 3))

        initial = manager.prepare_store([hot, cold])
        self.assertIsNotNone(initial)
        manager.complete_store([hot, cold])
        for _ in range(4):
            manager.touch([hot])
        manager.touch([cold])

        replacement = manager.prepare_store([incoming])

        self.assertEqual(replacement.block_hashes_evicted, [cold])
        self.assertIn(hot, manager.blocks)
        self.assertIn(incoming, manager.blocks)
        self.assertNotIn(cold, manager.blocks)
        self.assertEqual(manager.eviction_count, 1)

    def test_busy_block_is_not_evicted(self):
        manager = HeatOffloadingManager(FakeBackend(capacity=1))
        resident, incoming = map(block_hash, (4, 5))
        manager.prepare_store([resident])
        manager.complete_store([resident])
        manager.prepare_load([resident])

        self.assertIsNone(manager.prepare_store([incoming]))
        self.assertIn(resident, manager.blocks)


if __name__ == "__main__":
    unittest.main()
