import numpy as np
import pandas as pd
import pytest

from charge_key_automation.pairwise_reranker import (
    PAIR_FEATURES,
    build_pairwise_training_data,
    evaluate_pairwise_reranker,
    fit_pairwise_reranker,
    rerank_query_candidates,
)
from charge_key_automation.retrieval_eval import RetrievalEvaluationConfig


def _adversarial_reference():
    return pd.DataFrame(
        {
            "chrg_code": ["A1", "B2", "C3", "D4", "E5", "F6"],
            "chrg_desc": [
                "COMMON THEFT",
                "COMMON ASSAULT",
                "COMMON ROBBERY",
                "COMMON BURGLARY",
                "COMMON FRAUD",
                "COMMON TRESPASS",
            ],
        },
        index=[1, 2, 3, 4, 5, 6],
    )


def _adversarial_queries():
    reference = _adversarial_reference()
    rows = []

    # Each query keeps the correct code but borrows the next row's description.
    # The default retrieval weights therefore prefer the wrong exact-description
    # row, while a learned pairwise model can discover that code similarity is
    # the reliable signal across different reference groups.
    indexes = list(reference.index)
    for position, target in enumerate(indexes):
        wrong = indexes[(position + 1) % len(indexes)]
        rows.append(
            {
                "synthetic_code": reference.loc[target, "chrg_code"],
                "synthetic_desc": reference.loc[wrong, "chrg_desc"],
                "reference_index": target,
            }
        )
    return pd.DataFrame(rows, index=[f"q{i}" for i in range(len(rows))])


def test_pairwise_training_examples_are_symmetric_and_balanced():
    queries = _adversarial_queries().iloc[:2]
    x_train, y_train = build_pairwise_training_data(
        queries,
        _adversarial_reference(),
        negatives_per_query=2,
    )

    assert x_train.shape == (8, len(PAIR_FEATURES))
    assert y_train.tolist() == [1, 0, 1, 0, 1, 0, 1, 0]

    for position in range(0, len(x_train), 2):
        np.testing.assert_allclose(x_train[position], -x_train[position + 1])


def test_fitted_reranker_exposes_finite_feature_weights():
    reranker = fit_pairwise_reranker(
        _adversarial_queries().iloc[:4],
        _adversarial_reference(),
        negatives_per_query=3,
    )

    assert set(reranker.coefficients) == set(PAIR_FEATURES)
    assert all(np.isfinite(value) for value in reranker.coefficients.values())


def test_reranking_preserves_candidate_set_and_produces_dense_ranks():
    reference = _adversarial_reference()
    queries = _adversarial_queries()

    reranker = fit_pairwise_reranker(
        queries.iloc[:4],
        reference,
        negatives_per_query=3,
    )

    reranked = rerank_query_candidates(
        queries.iloc[4]["synthetic_code"],
        queries.iloc[4]["synthetic_desc"],
        reference,
        reranker,
    )

    assert set(reranked["candidate_index"]) == set(reference.index)
    assert reranked["rank"].tolist() == list(range(1, len(reference) + 1))
    assert "baseline_rank" in reranked.columns
    assert "reranker_score" in reranked.columns


def test_grouped_evaluation_prevents_synthetic_sibling_leakage():
    base = _adversarial_queries()
    queries = pd.concat([base, base], ignore_index=True)

    result = evaluate_pairwise_reranker(
        queries,
        _adversarial_reference(),
        test_size=0.34,
        random_state=7,
        negatives_per_query=3,
    )

    assert set(result.train_reference_indexes).isdisjoint(
        set(result.test_reference_indexes)
    )
    assert result.summary["n_train_reference_groups"] + result.summary[
        "n_test_reference_groups"
    ] == 6


def test_pairwise_reranker_improves_adversarial_held_out_ranking():
    result = evaluate_pairwise_reranker(
        _adversarial_queries(),
        _adversarial_reference(),
        test_size=0.34,
        random_state=42,
        negatives_per_query=3,
        retrieval_config=RetrievalEvaluationConfig(top_ks=(1, 3, 5)),
    )

    assert result.summary["baseline_recall_at_1"] == 0.0
    assert result.summary["reranked_recall_at_1"] == 1.0
    assert result.summary["mrr_delta"] > 0
    assert result.summary["improved_queries"] == result.summary["n_test_queries"]
    assert result.summary["worsened_queries"] == 0


def test_evaluation_is_deterministic_for_fixed_split_seed():
    kwargs = dict(
        queries=_adversarial_queries(),
        reference=_adversarial_reference(),
        test_size=0.34,
        random_state=19,
        negatives_per_query=3,
    )

    left = evaluate_pairwise_reranker(**kwargs)
    right = evaluate_pairwise_reranker(**kwargs)

    assert left.summary == right.summary
    assert left.coefficients == right.coefficients
    pd.testing.assert_frame_equal(left.per_query, right.per_query)


def test_evaluation_requires_multiple_reference_groups():
    queries = _adversarial_queries().iloc[[0]].copy()

    with pytest.raises(ValueError, match="at least two distinct reference groups"):
        evaluate_pairwise_reranker(
            queries,
            _adversarial_reference(),
        )
