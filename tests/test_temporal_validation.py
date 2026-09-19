import math

import pandas as pd
import pytest

from charge_key_automation.temporal_validation import (
    TemporalValidationConfig,
    evaluate_temporal_holdout,
    fit_temporal_reranker,
)


def _history():
    return pd.DataFrame(
        {
            "chrg_code": ["A1", "B2", "C3", "D4", "E5", "F6"],
            "chrg_desc": [
                "AGGRAVATED ASSAULT WITH DEADLY WEAPON",
                "POSSESSION OF CONTROLLED SUBSTANCE",
                "SIMPLE ASSAULT",
                "DOMESTIC VIOLENCE ASSAULT",
                "FRAUD BY DECEPTION",
                "TRESPASS PROPERTY",
            ],
            "source_year": [2021, 2022, 2023, 2024, 2025, 2026],
        },
        index=[10, 20, 30, 40, 50, 60],
    )


def _real_2025_queries():
    return pd.DataFrame(
        {
            "chrg_code": ["A1", "B2", "NEW", "E5"],
            "chrg_desc": [
                "AGG ASSLT W DEADLY WPN",
                "POSS CONTROLLED SUBSTANCE",
                "NEW OFFENSE NEVER SEEN BEFORE",
                "FRAUD BY DECEPTION",
            ],
            "source_year": [2025, 2025, 2025, 2025],
            # 10 and 20 are verified historical matches.
            # Missing and out-of-reference targets are novel/unscorable.
            "reference_index": [10, 20, None, 50],
        },
        index=["mapped_a", "mapped_b", "novel_missing", "novel_2025_only"],
    )


def test_temporal_training_excludes_validation_and_future_years():
    config = TemporalValidationConfig(
        validation_year=2025,
        variants_per_row=2,
        synthetic_seed=7,
        negatives_per_query=2,
    )

    _, training_reference, synthetic_training = fit_temporal_reranker(
        _history(),
        config=config,
    )

    assert training_reference.index.tolist() == [10, 20, 30, 40]
    assert set(training_reference["source_year"]) == {2021, 2022, 2023, 2024}
    assert len(synthetic_training) == 8
    assert set(synthetic_training["reference_index"]) == {10, 20, 30, 40}


def test_real_holdout_reports_mapped_and_novel_rows_separately():
    result = evaluate_temporal_holdout(
        _history(),
        _real_2025_queries(),
        config=TemporalValidationConfig(
            validation_year=2025,
            variants_per_row=2,
            negatives_per_query=2,
        ),
    )

    assert result.summary["n_validation_queries"] == 4
    assert result.summary["n_mapped_queries"] == 2
    assert result.summary["n_novel_queries"] == 2
    assert result.summary["mapped_coverage"] == 0.5

    status = result.per_query.set_index("query_index")["is_mapped"].to_dict()
    assert bool(status["mapped_a"]) is True
    assert bool(status["mapped_b"]) is True
    assert bool(status["novel_missing"]) is False
    assert bool(status["novel_2025_only"]) is False

    assert set(result.training_reference_indexes) == {10, 20, 30, 40}


def test_holdout_metrics_are_computed_only_for_verified_mapped_rows():
    result = evaluate_temporal_holdout(
        _history(),
        _real_2025_queries(),
        config=TemporalValidationConfig(
            variants_per_row=2,
            negatives_per_query=2,
        ),
    )

    mapped = result.per_query[result.per_query["is_mapped"]]
    assert mapped["baseline_rank"].notna().all()
    assert mapped["reranked_rank"].notna().all()

    novel = result.per_query[~result.per_query["is_mapped"]]
    assert novel["baseline_rank"].isna().all()
    assert novel["reranked_rank"].isna().all()

    assert 0.0 <= result.summary["baseline_top1_accuracy"] <= 1.0
    assert 0.0 <= result.summary["reranked_top1_accuracy"] <= 1.0


def test_missing_ground_truth_column_is_rejected_instead_of_inferred():
    validation = _real_2025_queries().drop(columns=["reference_index"])

    with pytest.raises(ValueError, match="explicit ground-truth target"):
        evaluate_temporal_holdout(
            _history(),
            validation,
            config=TemporalValidationConfig(
                variants_per_row=2,
                negatives_per_query=2,
            ),
        )


def test_validation_rows_must_belong_to_declared_validation_year():
    validation = _real_2025_queries().copy()
    validation.loc["mapped_a", "source_year"] = 2024

    with pytest.raises(ValueError, match="outside validation_year"):
        evaluate_temporal_holdout(
            _history(),
            validation,
            config=TemporalValidationConfig(
                validation_year=2025,
                variants_per_row=2,
                negatives_per_query=2,
            ),
        )


def test_zero_margin_accepts_all_and_counts_novel_false_accepts():
    result = evaluate_temporal_holdout(
        _history(),
        _real_2025_queries(),
        config=TemporalValidationConfig(
            variants_per_row=2,
            negatives_per_query=2,
            acceptance_margin=0.0,
        ),
    )

    assert result.summary["auto_match_count"] == 4
    assert result.summary["review_count"] == 0
    assert result.summary["review_rate"] == 0.0
    assert result.summary["novel_auto_accept_count"] == 2


def test_extreme_margin_reviews_all_including_novel_rows():
    result = evaluate_temporal_holdout(
        _history(),
        _real_2025_queries(),
        config=TemporalValidationConfig(
            variants_per_row=2,
            negatives_per_query=2,
            acceptance_margin=1e9,
        ),
    )

    assert result.summary["auto_match_count"] == 0
    assert result.summary["review_count"] == 4
    assert result.summary["review_rate"] == 1.0
    assert result.summary["novel_auto_accept_count"] == 0
    assert result.summary["novel_review_rate"] == 1.0
    assert math.isnan(result.summary["auto_match_precision"])


def test_non_numeric_history_years_are_rejected():
    history = _history()
    history["source_year"] = history["source_year"].astype(object)
    history.loc[10, "source_year"] = "unknown"

    with pytest.raises(ValueError, match="non-numeric or missing"):
        fit_temporal_reranker(
            history,
            config=TemporalValidationConfig(
                variants_per_row=2,
                negatives_per_query=2,
            ),
        )


def test_temporal_validation_requires_multiple_training_reference_rows():
    history = _history().loc[[10, 50]]

    with pytest.raises(ValueError, match="at least two pre-validation"):
        fit_temporal_reranker(
            history,
            config=TemporalValidationConfig(
                variants_per_row=2,
                negatives_per_query=1,
            ),
        )


def test_empty_real_holdout_is_rejected():
    empty = _real_2025_queries().iloc[0:0]

    with pytest.raises(ValueError, match="at least one real observation"):
        evaluate_temporal_holdout(
            _history(),
            empty,
            config=TemporalValidationConfig(
                variants_per_row=2,
                negatives_per_query=2,
            ),
        )
