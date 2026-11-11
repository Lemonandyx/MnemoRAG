import json
import os
from array import array
from collections import Counter, OrderedDict
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from threading import Lock
from typing import Hashable, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(slots=True)
class QueryState:
    """Historical retrieval state retained in the Query Base."""

    query_id: Optional[str]
    query_text: Optional[str]
    embedding: Optional[torch.Tensor]
    topk_doc_ids: Optional[Sequence[int]] = None
    topk_scores: Optional[Sequence[float]] = None
    topk_cluster_ids: Optional[Sequence[int]] = None


class QueryPool:
    """Capacity-bounded Query Base backed by a contiguous embedding matrix."""

    def __init__(
        self,
        device="cuda",
        dtype=torch.float16,
        max_size=80_000,
        replacement_policy="lru",
        retain_query_metadata=False,
        enable_exact_lookup=False,
    ):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if replacement_policy != "lru":
            raise ValueError("QueryPool currently supports only LRU replacement")
        if isinstance(dtype, str):
            dtype_map = {
                "fp16": torch.float16,
                "float16": torch.float16,
                "fp32": torch.float32,
                "float32": torch.float32,
            }
            if dtype not in dtype_map:
                raise ValueError(f"Unsupported QueryPool dtype: {dtype}")
            dtype = dtype_map[dtype]

        self.device = torch.device(device)
        self.dtype = dtype
        self.max_size = int(max_size)
        self.replacement_policy = replacement_policy
        self.retain_query_metadata = bool(retain_query_metadata)
        self.enable_exact_lookup = bool(enable_exact_lookup)
        self.query_states: list[QueryState] = []
        self.embedding_matrix: Optional[torch.Tensor] = None
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._exact_query_to_slot: dict[str, int] = {}
        self._slot_exact_queries: list[Optional[str]] = []
        self.min_similarity = 1
        self.max_similarity = 0

    def current_size(self):
        return len(self.query_states)

    def _initialize_matrix(self, embedding_dim: int) -> None:
        if self.embedding_matrix is None:
            self.embedding_matrix = torch.empty(
                (self.max_size, embedding_dim),
                device=self.device,
                dtype=self.dtype,
            )
        elif self.embedding_matrix.shape[1] != embedding_dim:
            raise ValueError(
                "Query embedding dimension changed from "
                f"{self.embedding_matrix.shape[1]} to {embedding_dim}"
            )

    def _select_insertion_slot(self) -> int:
        if self.current_size() < self.max_size:
            return self.current_size()
        slot, _ = self._lru.popitem(last=False)
        return slot

    def _mark_recent(self, slot: int) -> None:
        self._lru.pop(slot, None)
        self._lru[slot] = None

    def _compact_state(self, state: QueryState) -> None:
        state.embedding = None
        if not self.retain_query_metadata:
            state.query_id = None
            state.query_text = None
        if state.topk_doc_ids is not None and not isinstance(state.topk_doc_ids, array):
            state.topk_doc_ids = array("q", map(int, state.topk_doc_ids))
        if state.topk_scores is not None and not isinstance(state.topk_scores, array):
            state.topk_scores = array("f", map(float, state.topk_scores))
        if state.topk_cluster_ids is not None and not isinstance(
            state.topk_cluster_ids, array
        ):
            state.topk_cluster_ids = array("i", map(int, state.topk_cluster_ids))

    def touch(self, slots) -> None:
        """Refresh entries that contributed to reuse or guided retrieval."""
        if isinstance(slots, torch.Tensor):
            slots = slots.detach().cpu().reshape(-1).tolist()
        elif isinstance(slots, np.ndarray):
            slots = slots.reshape(-1).tolist()

        for slot in slots:
            slot = int(slot)
            if 0 <= slot < self.current_size():
                self._mark_recent(slot)

    def lookup_exact_batch(self, queries: Sequence[str]):
        """Return Query Base slots for exact strings without changing LRU state."""
        if not self.enable_exact_lookup:
            return None
        return [self._exact_query_to_slot.get(query) for query in queries]

    def _replace_exact_mapping(self, slot: int, query_text: Optional[str]) -> None:
        if slot < len(self._slot_exact_queries):
            previous = self._slot_exact_queries[slot]
            if self._exact_query_to_slot.get(previous) == slot:
                self._exact_query_to_slot.pop(previous, None)
        else:
            self._slot_exact_queries.append(None)

        exact_query = query_text if self.enable_exact_lookup else None
        if exact_query is not None:
            previous_slot = self._exact_query_to_slot.get(exact_query)
            if previous_slot is not None and previous_slot != slot:
                self._slot_exact_queries[previous_slot] = None
            self._exact_query_to_slot[exact_query] = slot
        self._slot_exact_queries[slot] = exact_query

    @torch.no_grad()
    def add_queries_batch(self, query_states: list[QueryState]):
        if not query_states:
            return
        if len(query_states) > self.max_size:
            query_states = query_states[-self.max_size :]

        embeddings = [state.embedding for state in query_states]
        if isinstance(embeddings[0], np.ndarray):
            embeddings = torch.from_numpy(np.stack(embeddings))
        else:
            embeddings = torch.stack(embeddings)
        embeddings = embeddings.to(self.device, dtype=self.dtype)
        embeddings = F.normalize(embeddings, dim=1)
        self._initialize_matrix(embeddings.shape[1])

        slots = []
        for state in query_states:
            existing_slot = None
            if self.enable_exact_lookup and state.query_text is not None:
                existing_slot = self._exact_query_to_slot.get(state.query_text)
            slot = (
                int(existing_slot)
                if existing_slot is not None
                else self._select_insertion_slot()
            )
            self._replace_exact_mapping(slot, state.query_text)
            if slot == self.current_size():
                self.query_states.append(state)
            else:
                self.query_states[slot] = state
            self._mark_recent(slot)
            slots.append(slot)

        slot_indices = torch.tensor(slots, device=self.device, dtype=torch.long)
        self.embedding_matrix.index_copy_(0, slot_indices, embeddings)
        for slot in slots:
            self._compact_state(self.query_states[slot])

    @staticmethod
    def _pack_optional_sequences(states, attribute, dtype, fill_value):
        sequences = [getattr(state, attribute) for state in states]
        lengths = np.asarray(
            [0 if values is None else len(values) for values in sequences],
            dtype="<i4",
        )
        width = int(lengths.max(initial=0))
        packed = np.full((len(states), width), fill_value, dtype=dtype)
        for index, values in enumerate(sequences):
            if values is not None and len(values):
                packed[index, : len(values)] = np.asarray(values, dtype=dtype)
        return packed, lengths

    def save_snapshot(self, path, *, metadata=None):
        """Persist a portable, pickle-free Query Base warm-start snapshot."""
        if self.current_size() == 0 or self.embedding_matrix is None:
            raise ValueError("cannot snapshot an empty QueryPool")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        size = self.current_size()
        embeddings = self.embedding_matrix[:size].detach().cpu().numpy().copy()
        doc_ids, doc_lengths = self._pack_optional_sequences(
            self.query_states, "topk_doc_ids", "<i8", -1
        )
        scores, score_lengths = self._pack_optional_sequences(
            self.query_states, "topk_scores", "<f4", np.nan
        )
        clusters, cluster_lengths = self._pack_optional_sequences(
            self.query_states, "topk_cluster_ids", "<i4", -1
        )
        exact_queries = list(self._slot_exact_queries[:size])
        exact_present = np.asarray(
            [value is not None for value in exact_queries], dtype=np.bool_
        )
        exact_values = np.asarray(
            ["" if value is None else value for value in exact_queries],
            dtype=np.str_,
        )
        snapshot_metadata = {
            "format": "mnemorag.query-base.v1",
            "entry_count": size,
            "embedding_dim": int(embeddings.shape[1]),
            "embedding_dtype": str(embeddings.dtype),
            "metadata": metadata or {},
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                embeddings=embeddings,
                topk_doc_ids=doc_ids,
                topk_doc_lengths=doc_lengths,
                topk_scores=scores,
                topk_score_lengths=score_lengths,
                topk_cluster_ids=clusters,
                topk_cluster_lengths=cluster_lengths,
                exact_query_values=exact_values,
                exact_query_present=exact_present,
                metadata_json=np.asarray(
                    json.dumps(
                        snapshot_metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ),
            )
        temporary.replace(path)
        return snapshot_metadata

    @staticmethod
    def _unpack_snapshot_sequence(values, lengths, index):
        length = int(lengths[index])
        return values[index, :length].tolist() if length else None

    def load_snapshot(self, path):
        """Load a Query Base snapshot before timed request processing begins."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as snapshot:
            metadata = json.loads(str(snapshot["metadata_json"].item()))
            if metadata.get("format") != "mnemorag.query-base.v1":
                raise ValueError("unsupported QueryPool snapshot format")
            embeddings = np.asarray(snapshot["embeddings"])
            doc_ids = np.asarray(snapshot["topk_doc_ids"], dtype="<i8")
            doc_lengths = np.asarray(snapshot["topk_doc_lengths"], dtype="<i4")
            scores = np.asarray(snapshot["topk_scores"], dtype="<f4")
            score_lengths = np.asarray(snapshot["topk_score_lengths"], dtype="<i4")
            clusters = np.asarray(snapshot["topk_cluster_ids"], dtype="<i4")
            cluster_lengths = np.asarray(snapshot["topk_cluster_lengths"], dtype="<i4")
            exact_values = np.asarray(snapshot["exact_query_values"], dtype=np.str_)
            exact_present = np.asarray(snapshot["exact_query_present"], dtype=np.bool_)

        count = int(metadata.get("entry_count", -1))
        if embeddings.ndim != 2 or embeddings.shape[0] != count:
            raise ValueError("QueryPool snapshot embedding shape mismatch")
        vectors = (doc_lengths, score_lengths, cluster_lengths, exact_present)
        if any(vector.shape != (count,) for vector in vectors):
            raise ValueError("QueryPool snapshot metadata shape mismatch")
        if exact_values.shape != (count,):
            raise ValueError("QueryPool snapshot query-text shape mismatch")
        if count > self.max_size:
            raise ValueError(
                f"snapshot has {count} entries but capacity is {self.max_size}"
            )
        if self.current_size() != 0:
            raise ValueError("QueryPool snapshot must be loaded into an empty pool")

        states = []
        for index in range(count):
            states.append(
                QueryState(
                    query_id=None,
                    query_text=(
                        str(exact_values[index]) if exact_present[index] else None
                    ),
                    embedding=embeddings[index],
                    topk_doc_ids=self._unpack_snapshot_sequence(
                        doc_ids, doc_lengths, index
                    ),
                    topk_scores=self._unpack_snapshot_sequence(
                        scores, score_lengths, index
                    ),
                    topk_cluster_ids=self._unpack_snapshot_sequence(
                        clusters, cluster_lengths, index
                    ),
                )
            )
        self.add_queries_batch(states)
        return metadata

    @torch.no_grad()
    def search_similar_batch(self, emb_batch, topk: int = 1):
        if self.current_size() == 0:
            return None, None

        if isinstance(emb_batch, np.ndarray):
            if not emb_batch.flags.writeable:
                emb_batch = np.array(emb_batch, copy=True)
            emb_batch = torch.from_numpy(emb_batch)
        if emb_batch.ndim == 1:
            emb_batch = emb_batch.unsqueeze(0)
        emb_batch = emb_batch.to(self.device, dtype=self.dtype)

        query_embeddings = F.normalize(emb_batch, dim=1)
        history = self.embedding_matrix[: self.current_size()]
        similarities = query_embeddings @ history.T
        k = min(topk, similarities.size(1))
        if k == 1:
            scores, indices = torch.max(similarities, dim=1, keepdim=True)
        else:
            scores, indices = torch.topk(similarities, k=k, dim=1)
        return scores.cpu(), indices.cpu()


class DocumentPool:
    def __init__(self, enable_hotness=True, num_documents=None):
        self.enable_hotness = enable_hotness
        self.num_documents = int(num_documents) if num_documents is not None else 0
        if self.num_documents < 0:
            raise ValueError("num_documents must be non-negative")
        self.dense_counts = (
            np.zeros(self.num_documents, dtype=np.uint32)
            if self.num_documents
            else None
        )
        self._dense_nonzero = 0
        self.doc_counter = Counter()
        self.pair_counter = {}
        pair_prior_path = os.environ.get("MNEMORAG_DOC_PAIR_HOTNESS_PRIOR_PATH", "")
        prior_path = os.environ.get("MNEMORAG_DOC_HOTNESS_PRIOR_PATH", "")
        self.freeze_prior = (
            bool(prior_path)
            and os.environ.get("MNEMORAG_DOC_HOTNESS_FREEZE_PRIOR", "0") == "1"
        )
        self.canonical_tie = (
            os.environ.get("MNEMORAG_HOTNESS_CANONICAL_TIE", "0") == "1"
        )
        if prior_path:
            with open(prior_path, "r", encoding="utf-8") as handle:
                raw_prior = json.load(handle)
            for doc_id, count in raw_prior.items():
                converted_id = int(doc_id) if str(doc_id).isdigit() else doc_id
                count = max(0, int(count))
                if self._is_dense_id(converted_id):
                    if count and self.dense_counts[converted_id] == 0:
                        self._dense_nonzero += 1
                    self.dense_counts[converted_id] = min(
                        count, np.iinfo(np.uint32).max
                    )
                elif count:
                    self.doc_counter[converted_id] = count
            print(
                f"Loaded document hotness prior: "
                f"{self.total_docs()} docs from {prior_path}"
            )
        if pair_prior_path:
            with open(pair_prior_path, "r", encoding="utf-8") as handle:
                raw_pairs = json.load(handle)

            def convert(value):
                return int(value) if str(value).isdigit() else value

            self.pair_counter = {
                convert(doc_id): {
                    convert(other_id): int(count)
                    for other_id, count in neighbors.items()
                }
                for doc_id, neighbors in raw_pairs.items()
            }
            print(
                f"Loaded document pair-hotness prior: "
                f"{len(self.pair_counter)} docs from {pair_prior_path}"
            )
        print(
            f"document hotness freeze_prior={self.freeze_prior}, "
            f"canonical_tie={self.canonical_tie}, "
            f"dense_count_bytes={self.dense_count_bytes}"
        )
        self.lock = Lock()

    @property
    def dense_count_bytes(self):
        return 0 if self.dense_counts is None else self.dense_counts.nbytes

    def _is_dense_id(self, doc_id):
        return (
            self.dense_counts is not None
            and isinstance(doc_id, Integral)
            and 0 <= int(doc_id) < self.num_documents
        )

    def update(self, doc_ids: Iterable[Hashable]):
        """Update exact document-access counts with bounded dense storage."""
        if self.freeze_prior:
            return
        doc_ids = list(doc_ids)
        if not doc_ids:
            return
        with self.lock:

            def update_dense(dense_array):
                unique_ids, increments = np.unique(dense_array, return_counts=True)
                current = self.dense_counts[unique_ids]
                self._dense_nonzero += int(np.count_nonzero(current == 0))
                updated = np.minimum(
                    current.astype(np.uint64) + increments.astype(np.uint64),
                    np.iinfo(np.uint32).max,
                )
                self.dense_counts[unique_ids] = updated.astype(np.uint32)

            candidate = np.asarray(doc_ids)
            if self.dense_counts is not None and candidate.dtype.kind in "iu":
                valid = (candidate >= 0) & (candidate < self.num_documents)
                if bool(np.all(valid)):
                    update_dense(candidate.astype(np.int64, copy=False))
                    return

            dense_ids = []
            fallback_ids = []
            for doc_id in doc_ids:
                if self._is_dense_id(doc_id):
                    dense_ids.append(int(doc_id))
                else:
                    fallback_ids.append(doc_id)
            if dense_ids:
                dense_array = np.asarray(dense_ids, dtype=np.int64)
                update_dense(dense_array)
            if fallback_ids:
                self.doc_counter.update(fallback_ids)

    def get_count(self, doc_id):
        if self._is_dense_id(doc_id):
            return int(self.dense_counts[int(doc_id)])
        return self.doc_counter.get(doc_id, 0)

    def topk(self, k=10):
        """Return the most frequently accessed document IDs."""
        if k <= 0:
            return []
        candidates = self.doc_counter.most_common(k)
        dense_k = min(k, self._dense_nonzero)
        if dense_k:
            indices = np.argpartition(self.dense_counts, -dense_k)[-dense_k:]
            candidates.extend(
                (int(index), int(self.dense_counts[index]))
                for index in indices
                if self.dense_counts[index]
            )
        candidates.sort(key=lambda item: (-item[1], str(item[0])))
        return candidates[:k]

    def total_docs(self):
        return self._dense_nonzero + len(self.doc_counter)

    def _order_equal_heat_ties(self, pairs):
        groups = {}
        for pair in pairs:
            groups.setdefault(self.get_count(pair[0]), []).append(pair)
        ordered = []
        for heat in sorted(groups, reverse=True):
            group = groups[heat]
            while group:
                if ordered:
                    previous = ordered[-1][0]

                    def score(pair):
                        return self.pair_counter.get(previous, {}).get(pair[0], 0)
                else:

                    def score(pair):
                        return max(
                            (
                                self.pair_counter.get(pair[0], {}).get(candidate[0], 0)
                                for candidate in pairs
                                if candidate[0] != pair[0]
                            ),
                            default=0,
                        )

                group.sort(key=lambda pair: (-score(pair), str(pair[0])))
                ordered.append(group.pop(0))
        return ordered

    def reorder_by_hotness(self, doc_idxs, doc_contents):
        """Order documents by access heat while preserving cold-start order."""
        if not self.enable_hotness:
            return doc_contents, doc_idxs

        counts = [self.get_count(i) for i in doc_idxs]

        if not counts or max(counts) <= 1:
            return doc_contents, doc_idxs

        pairs = list(zip(doc_idxs, doc_contents))

        if self.canonical_tie:
            if self.pair_counter:
                pairs = self._order_equal_heat_ties(pairs)
            else:
                pairs.sort(key=lambda x: (-self.get_count(x[0]), str(x[0])))
        else:
            pairs.sort(key=lambda x: self.get_count(x[0]), reverse=True)

        reordered_idxs, reordered_contents = zip(*pairs)
        return list(reordered_contents), list(reordered_idxs)
