import numpy as np
import pytest

from benchmarks.prepare_disjoint_history import (
    build_hotness_prior,
    merge_history_into_profile,
)


def test_hotness_prior_counts_only_repeated_top3_documents():
    ids = np.asarray([[1, 2, 3], [1, 4, 5], [1, 2, 6]], dtype=np.int64)

    assert build_hotness_prior(ids, minimum_frequency=2) == {"1": 3, "2": 2}


def test_hotness_prior_rejects_non_top3_or_invalid_ids():
    with pytest.raises(ValueError, match="shape"):
        build_hotness_prior(np.asarray([[1, 2]]), minimum_frequency=1)
    with pytest.raises(ValueError, match="invalid"):
        build_hotness_prior(np.asarray([[1, 2, -1]]), minimum_frequency=1)


def test_profile_merge_binds_both_history_artifacts():
    history = {
        "prior_path": "/history/prior.json",
        "query_base_snapshot_path": "/history/query-base.npz",
    }
    profile = {
        "version": 1,
        "extra_args": [],
        "environment": {},
        "dataset_environment": {},
    }

    merged = merge_history_into_profile(profile, "nq", history)

    assert merged["environment"]["MNEMORAG_DOC_HOTNESS_FREEZE_PRIOR"] == "1"
    assert merged["dataset_history"]["nq"] == history
    assert (
        merged["dataset_environment"]["nq"]["MNEMORAG_QUERY_BASE_SNAPSHOT_PATH"]
        == "/history/query-base.npz"
    )
    assert profile["environment"] == {}
