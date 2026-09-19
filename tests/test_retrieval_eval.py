import pandas as pd
import pytest

from charge_key_automation.retrieval_eval import (
    RetrievalEvaluationConfig,
    evaluate_candidate_retrieval,
    mine_hard_negatives,
    rank_query_candidates,
)


def _reference():
    return pd.DataFrame(
        {
            "chrg_code": ["14-33", "14-33A", "90-95", "20-138"],
            "chrg_desc": [
                "AGGRAVATED ASSAULT WITH DEADLY WEAPON",
                "AGGRAVATED ASSAULT WITH FIREARM",
                "POSSESSION OF CONTROLLED SUBSTANCE",
                "SIMPLE ASSAULT",
            ],
        },
        index=[11, 22, 33, 44],
    )


def _queries():
    return pd.DataFrame(
        {
            "synthetic_code": ["14 33", "90.95"],
            "synthetic_desc": [
                "AGG ASSLT W/ DEADLY WPN",
                "POSS CTRL SUBST",
            ],
            "reference_index": [11, 33],
        },
        index=["q1", "q2"],
    )


def test_noisy_queries_retrieve_their_reference_rows_at_top_one():
    result = evaluate_candidate_retrieval(
        _queries(),
        _reference(),
        config=RetrievalEvaluationConfig(top_ks=(1, 3, 5), hard_negatives_per_query=2),
    )

    assert result.summary["n_queries"] == 2
    assert result.summary["recall_at_1"] == 1.0
    assert result.summary["recall_at_3"] == 1.0
    assert result.summary["recall_at_5"] == 1.0
    assert result.summary["mrr"] == 1.0
    assert result.per_query["rank"].tolist() == [1, 1]


def test_hard_negatives_never_include_the_ground_truth_row():
    result = evaluate_candidate_retrieval(
        _queries(),
        _reference(),
        config=RetrievalEvaluationConfig(hard_negatives_per_query=2),
    )

    assert len(result.hard_negatives) == 4
    assert (
        result.hard_negatives["negative_index"]
        != result.hard_negatives["reference_index"]
    ).all()


def test_hard_negative_mining_is_deterministic():
    config = RetrievalEvaluationConfig(hard_negatives_per_query=3)

    left = mine_hard_negatives(_queries(), _reference(), config=config)
    right = mine_hard_negatives(_queries(), _reference(), config=config)

    pd.testing.assert_frame_equal(left, right)


def test_tie_breaking_produces_measurable_top_k_behavior():
    reference = pd.DataFrame(
        {
            "chrg_code": ["", ""],
            "chrg_desc": ["FAILURE TO APPEAR", "FAILURE TO APPEAR"],
        },
        index=[5, 9],
    )
    queries = pd.DataFrame(
        {
            "synthetic_code": [""],
            "synthetic_desc": ["FAILURE TO APPEAR"],
            "reference_index": [9],
        },
        index=["ambiguous"],
    )

    result = evaluate_candidate_retrieval(
        queries,
        reference,
        config=RetrievalEvaluationConfig(top_ks=(1, 2), hard_negatives_per_query=1),
    )

    assert result.per_query.loc[0, "rank"] == 2
    assert result.summary["recall_at_1"] == 0.0
    assert result.summary["recall_at_2"] == 1.0
    assert result.summary["mrr"] == 0.5


def test_single_query_ranking_exposes_score_components():
    ranked = rank_query_candidates(
        "14.33",
        "AGGRAVATED ASSAULT DEADLY WEAPON",
        _reference(),
    )

    expected = {
        "rank",
        "candidate_index",
        "retrieval_score",
        "code_similarity",
        "token_sort_similarity",
        "token_set_similarity",
        "char_tfidf_similarity",
    }
    assert expected.issubset(ranked.columns)
    assert ranked.iloc[0]["candidate_index"] == 11
    assert ranked["retrieval_score"].is_monotonic_decreasing


def test_custom_reference_column_names_are_supported():
    reference = _reference().rename(
        columns={"chrg_code": "code", "chrg_desc": "description"}
    )
    queries = _queries().rename(
        columns={
            "synthetic_code": "query_code",
            "synthetic_desc": "query_description",
            "reference_index": "target",
        }
    )

    result = evaluate_candidate_retrieval(
        queries,
        reference,
        query_code_col="query_code",
        query_desc_col="query_description",
        target_col="target",
        ref_code_col="code",
        ref_desc_col="description",
    )

    assert result.summary["recall_at_1"] == 1.0


def test_unknown_reference_target_is_rejected():
    queries = _queries().copy()
    queries.loc["q1", "reference_index"] = 999

    with pytest.raises(ValueError, match="absent from reference"):
        evaluate_candidate_retrieval(queries, _reference())


def test_reference_index_must_be_unique():
    reference = _reference()
    reference.index = [1, 1, 2, 3]

    with pytest.raises(ValueError, match="index must be unique"):
        evaluate_candidate_retrieval(_queries(), reference)


def test_all_blank_reference_descriptions_preserve_candidate_alignment():
    reference = pd.DataFrame(
        {
            "chrg_code": ["A1", "B2", "C3"],
            "chrg_desc": ["", "", ""],
        },
        index=[10, 20, 30],
    )

    ranked = rank_query_candidates("B2", "", reference)

    assert len(ranked) == len(reference)
    assert set(ranked["candidate_index"]) == {10, 20, 30}
    assert ranked.iloc[0]["candidate_index"] == 20
