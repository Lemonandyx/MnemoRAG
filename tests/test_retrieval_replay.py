import csv
import json

import pytest

from benchmarks.benchmark_point import (
    install_retrieval_replay,
    load_retrieval_replay_csv,
)


class FakeRetriever:
    def __init__(self):
        self.topk = 1
        self.corpus = [
            {"id": "0", "title": "zero", "text": "alpha"},
            {"id": "1", "title": "one", "text": "beta"},
        ]


def write_replay(path):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "request_index",
                "question",
                "retrieval_idxs",
                "retrieval_scores",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "request_index": 0,
                "question": "q0",
                "retrieval_idxs": json.dumps([1]),
                "retrieval_scores": json.dumps([0.9]),
            }
        )
        writer.writerow(
            {
                "request_index": 1,
                "question": "q1",
                "retrieval_idxs": json.dumps([0]),
                "retrieval_scores": json.dumps([0.8]),
            }
        )


def test_retrieval_replay_preserves_request_order_and_ids(tmp_path):
    path = tmp_path / "requests.csv"
    write_replay(path)
    replay = load_retrieval_replay_csv(path, expected_count=2, expected_topk=1)
    retriever = FakeRetriever()
    state = install_retrieval_replay(
        retriever,
        replay,
        lambda corpus, idxs: [corpus[idx] for idx in idxs],
    )

    docs, scores, idxs = retriever.batch_search(
        ["q0"], return_score=True, return_idx=True
    )
    assert idxs == [[1]]
    assert scores == [[0.9]]
    assert docs[0][0]["text"] == "beta"
    assert state["cursor"] == 1

    docs, scores, idxs = retriever.batch_search(
        ["q1"], return_score=True, return_idx=True
    )
    assert idxs == [[0]]
    assert docs[0][0]["text"] == "alpha"
    assert state["cursor"] == 2
    assert retriever.last_request_actions == ["replay"]


def test_retrieval_replay_rejects_query_drift(tmp_path):
    path = tmp_path / "requests.csv"
    write_replay(path)
    replay = load_retrieval_replay_csv(path, expected_count=2, expected_topk=1)
    retriever = FakeRetriever()
    install_retrieval_replay(
        retriever,
        replay,
        lambda corpus, idxs: [corpus[idx] for idx in idxs],
    )

    with pytest.raises(RuntimeError, match="query mismatch"):
        retriever.batch_search(["wrong"], return_score=True, return_idx=True)
