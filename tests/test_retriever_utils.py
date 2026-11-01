import pytest
from datasets import Dataset

from mnemorag.retriever.utils import adaptive_probe_budget, load_docs


def test_load_docs_batches_huggingface_dataset_and_preserves_order():
    corpus = Dataset.from_dict(
        {
            "id": ["0", "1", "2"],
            "contents": ["zero", "one", "two"],
        }
    )

    assert load_docs(corpus, [2, 0, 2]) == [
        {"id": "2", "contents": "two"},
        {"id": "0", "contents": "zero"},
        {"id": "2", "contents": "two"},
    ]


def test_load_docs_preserves_generic_sequence_support():
    corpus = [{"id": "0"}, {"id": "1"}]

    assert load_docs(corpus, [1, 0]) == [{"id": "1"}, {"id": "0"}]
    assert load_docs(corpus, []) == []


@pytest.mark.parametrize("invalid_id", [-1, 2])
def test_load_docs_rejects_out_of_bounds_document_ids(invalid_id):
    corpus = [{"id": "0"}, {"id": "1"}]

    with pytest.raises(IndexError, match="corpus bounds"):
        load_docs(corpus, [invalid_id])


def test_adaptive_probe_budget_is_threshold_anchored_and_monotonic():
    scores = [0.70, 0.75, 0.80, 0.90, 0.95]
    budgets = [
        adaptive_probe_budget(score, 0.70, 0.95, 16, 64, gamma=2.0) for score in scores
    ]

    assert budgets == [64, 47, 34, 18, 16]
    assert all(left >= right for left, right in zip(budgets, budgets[1:]))


def test_adaptive_probe_budget_clamps_scores_and_power_of_two_rounding():
    assert adaptive_probe_budget(0.10, 0.70, 0.95, 16, 64) == 64
    assert adaptive_probe_budget(1.00, 0.70, 0.95, 16, 64) == 16
    assert (
        adaptive_probe_budget(
            0.80,
            0.70,
            0.95,
            16,
            64,
            gamma=2.0,
            rounding="power_of_two",
        )
        == 64
    )


def test_late_decay_preserves_budget_until_similarity_is_high():
    scores = [0.70, 0.75, 0.80, 0.90, 0.95]
    budgets = [
        adaptive_probe_budget(
            score,
            0.70,
            0.95,
            16,
            64,
            gamma=2.0,
            curve="late_decay",
        )
        for score in scores
    ]

    assert budgets == [64, 63, 57, 34, 16]
    assert all(left >= right for left, right in zip(budgets, budgets[1:]))


def test_adaptive_probe_budget_rejects_unknown_curve():
    with pytest.raises(ValueError, match="similarity_nprobe_curve"):
        adaptive_probe_budget(0.80, 0.70, 0.95, 16, 64, curve="unknown")
