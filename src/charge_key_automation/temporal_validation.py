"""Leakage-safe temporal holdout evaluation for real charge observations.

This module fits the experimental reranker only from pre-validation-year
historical rows and synthetic perturbations of those rows. It then evaluates
real validation-year observations without inferring their ground-truth labels.

A validation query may have a verified reference_index pointing to a historical
pre-validation row. Missing or out-of-reference targets are treated as novel /
unscorable rows and are reported separately rather than force-matched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .pairwise_reranker import (
    PairwiseReranker,
    fit_pairwise_reranker,
    rerank_query_candidates,
)
from .retrieval_eval import RetrievalEvaluationConfig, rank_query_candidates
from .synthetic import SyntheticAugmentationConfig, generate_synthetic_training_data


@dataclass(frozen=True)
class TemporalValidationConfig:
    """Configuration for a real out-of-time validation run."""

    validation_year: int = 2025
    year_col: str = "source_year"
    variants_per_row: int = 4
    synthetic_seed: int = 42
    negatives_per_query: int = 5
    regularization_c: float = 1.0
    random_state: int = 42
    acceptance_margin: float | None = None
    retrieval_config: RetrievalEvaluationConfig = field(
        default_factory=RetrievalEvaluationConfig
    )

    def __post_init__(self) -> None:
        if self.variants_per_row < 1:
            raise ValueError("variants_per_row must be at least 1")
        if self.negatives_per_query < 1:
            raise ValueError("negatives_per_query must be at least 1")
        if self.regularization_c <= 0:
            raise ValueError("regularization_c must be positive")
        if self.acceptance_margin is not None and self.acceptance_margin < 0:
            raise ValueError("acceptance_margin must be non-negative")


@dataclass
class TemporalValidationResult:
    """Real-year validation outputs."""

    summary: dict[str, float | int]
    per_query: pd.DataFrame
    coefficients: dict[str, float]
    training_reference_indexes: tuple[object, ...]


def _prepare_training_reference(
    reference_history: pd.DataFrame,
    config: TemporalValidationConfig,
) -> pd.DataFrame:
    required = {"chrg_code", "chrg_desc", config.year_col}
    missing = required - set(reference_history.columns)
    if missing:
        raise ValueError(
            f"reference_history is missing required columns: {sorted(missing)}"
        )
    if reference_history.index.has_duplicates:
        raise ValueError("reference_history index must be unique")

    years = pd.to_numeric(reference_history[config.year_col], errors="coerce")
    if years.isna().any():
        bad = reference_history.index[years.isna()].tolist()[:5]
        raise ValueError(
            f"{config.year_col!r} contains non-numeric or missing values at rows {bad}"
        )

    training = reference_history.loc[years < config.validation_year].copy()
    if len(training) < 2:
        raise ValueError(
            "at least two pre-validation reference rows are required for pairwise training"
        )
    return training


def _validate_holdout(
    validation_queries: pd.DataFrame,
    config: TemporalValidationConfig,
    *,
    query_code_col: str,
    query_desc_col: str,
    target_col: str,
) -> None:
    required = {query_code_col, query_desc_col, target_col}
    missing = required - set(validation_queries.columns)
    if missing:
        raise ValueError(
            "validation_queries must contain real observations plus an explicit "
            f"ground-truth target column; missing: {sorted(missing)}"
        )

    if config.year_col in validation_queries.columns:
        years = pd.to_numeric(validation_queries[config.year_col], errors="coerce")
        if years.isna().any():
            raise ValueError(
                f"validation query {config.year_col!r} values must be numeric"
            )
        wrong = validation_queries.loc[years != config.validation_year]
        if len(wrong):
            sample = wrong.index.tolist()[:5]
            raise ValueError(
                f"validation_queries contains rows outside validation_year="
                f"{config.validation_year}: {sample}"
            )


def fit_temporal_reranker(
    reference_history: pd.DataFrame,
    *,
    config: TemporalValidationConfig | None = None,
) -> tuple[PairwiseReranker, pd.DataFrame, pd.DataFrame]:
    """Fit the reranker using only pre-validation history and synthetic variants."""

    config = config or TemporalValidationConfig()
    training_reference = _prepare_training_reference(reference_history, config)

    synthetic_training = generate_synthetic_training_data(
        training_reference,
        year_col=config.year_col,
        config=SyntheticAugmentationConfig(
            variants_per_row=config.variants_per_row,
            seed=config.synthetic_seed,
        ),
    )

    if synthetic_training.empty:
        raise ValueError("synthetic training generation produced no usable queries")

    reranker = fit_pairwise_reranker(
        synthetic_training,
        training_reference,
        retrieval_config=config.retrieval_config,
        negatives_per_query=config.negatives_per_query,
        regularization_c=config.regularization_c,
        random_state=config.random_state,
    )
    return reranker, training_reference, synthetic_training


def _mapped_metrics(
    per_query: pd.DataFrame,
    *,
    top_ks: tuple[int, ...],
) -> dict[str, float | int]:
    mapped = per_query[per_query["is_mapped"]].copy()
    result: dict[str, float | int] = {}

    if mapped.empty:
        result.update(
            {
                "baseline_top1_accuracy": float("nan"),
                "reranked_top1_accuracy": float("nan"),
                "baseline_mrr": float("nan"),
                "reranked_mrr": float("nan"),
                "mrr_delta": float("nan"),
                "baseline_mean_rank": float("nan"),
                "reranked_mean_rank": float("nan"),
                "mean_rank_delta": float("nan"),
                "improved_queries": 0,
                "worsened_queries": 0,
                "unchanged_queries": 0,
            }
        )
        for k in sorted(set(top_ks)):
            result[f"baseline_recall_at_{k}"] = float("nan")
            result[f"reranked_recall_at_{k}"] = float("nan")
            result[f"recall_at_{k}_delta"] = float("nan")
        return result

    baseline_rank = mapped["baseline_rank"].astype(int)
    reranked_rank = mapped["reranked_rank"].astype(int)

    result["baseline_top1_accuracy"] = float((baseline_rank == 1).mean())
    result["reranked_top1_accuracy"] = float((reranked_rank == 1).mean())
    result["baseline_mrr"] = float((1.0 / baseline_rank).mean())
    result["reranked_mrr"] = float((1.0 / reranked_rank).mean())
    result["mrr_delta"] = float(result["reranked_mrr"] - result["baseline_mrr"])
    result["baseline_mean_rank"] = float(baseline_rank.mean())
    result["reranked_mean_rank"] = float(reranked_rank.mean())
    result["mean_rank_delta"] = float(
        result["reranked_mean_rank"] - result["baseline_mean_rank"]
    )

    for k in sorted(set(top_ks)):
        baseline = float((baseline_rank <= k).mean())
        reranked = float((reranked_rank <= k).mean())
        result[f"baseline_recall_at_{k}"] = baseline
        result[f"reranked_recall_at_{k}"] = reranked
        result[f"recall_at_{k}_delta"] = reranked - baseline

    result["improved_queries"] = int((reranked_rank < baseline_rank).sum())
    result["worsened_queries"] = int((reranked_rank > baseline_rank).sum())
    result["unchanged_queries"] = int((reranked_rank == baseline_rank).sum())
    return result


def _acceptance_metrics(
    per_query: pd.DataFrame,
    acceptance_margin: float,
) -> dict[str, float | int]:
    accepted = per_query["reranker_margin"] >= acceptance_margin
    reviewed = ~accepted

    accepted_rows = per_query[accepted]
    accepted_correct = (
        accepted_rows["is_mapped"]
        & accepted_rows["reranked_top_is_correct"]
    )

    novel = ~per_query["is_mapped"]
    novel_count = int(novel.sum())
    novel_reviewed = int((novel & reviewed).sum())

    auto_match_count = int(accepted.sum())
    precision = (
        float(accepted_correct.mean())
        if auto_match_count
        else float("nan")
    )

    return {
        "acceptance_margin": float(acceptance_margin),
        "auto_match_count": auto_match_count,
        "review_count": int(reviewed.sum()),
        "review_rate": float(reviewed.mean()),
        "auto_match_precision": precision,
        "novel_auto_accept_count": int((novel & accepted).sum()),
        "novel_review_rate": (
            float(novel_reviewed / novel_count)
            if novel_count
            else float("nan")
        ),
    }


def evaluate_temporal_holdout(
    reference_history: pd.DataFrame,
    validation_queries: pd.DataFrame,
    *,
    config: TemporalValidationConfig | None = None,
    query_code_col: str = "chrg_code",
    query_desc_col: str = "chrg_desc",
    target_col: str = "reference_index",
) -> TemporalValidationResult:
    """Evaluate the experimental reranker on real out-of-time observations.

    Ground truth is never inferred by the matcher under evaluation. The caller
    must provide target_col explicitly. Null targets and targets that do not
    exist in the pre-validation reference pool are treated as novel rows.
    """

    config = config or TemporalValidationConfig()
    _validate_holdout(
        validation_queries,
        config,
        query_code_col=query_code_col,
        query_desc_col=query_desc_col,
        target_col=target_col,
    )

    reranker, training_reference, synthetic_training = fit_temporal_reranker(
        reference_history,
        config=config,
    )

    rows: list[dict[str, object]] = []

    for query_index, query in validation_queries.iterrows():
        baseline = rank_query_candidates(
            query[query_code_col],
            query[query_desc_col],
            training_reference,
            config=config.retrieval_config,
        )
        reranked = rerank_query_candidates(
            query[query_code_col],
            query[query_desc_col],
            training_reference,
            reranker,
            retrieval_config=config.retrieval_config,
        )

        target = query[target_col]
        target_is_present = not pd.isna(target)
        is_mapped = bool(target_is_present and target in training_reference.index)

        baseline_top = baseline.iloc[0]
        reranked_top = reranked.iloc[0]
        second_reranker_score = (
            float(reranked.iloc[1]["reranker_score"])
            if len(reranked) > 1
            else float("-inf")
        )
        reranker_margin = (
            float(reranked_top["reranker_score"]) - second_reranker_score
            if np.isfinite(second_reranker_score)
            else float("inf")
        )

        if is_mapped:
            baseline_target = baseline[baseline["candidate_index"] == target].iloc[0]
            reranked_target = reranked[reranked["candidate_index"] == target].iloc[0]
            baseline_rank: float | int = int(baseline_target["rank"])
            reranked_rank: float | int = int(reranked_target["rank"])
        else:
            baseline_rank = float("nan")
            reranked_rank = float("nan")

        rows.append(
            {
                "query_index": query_index,
                "reference_index": target,
                "is_mapped": is_mapped,
                "is_novel": not is_mapped,
                "baseline_rank": baseline_rank,
                "reranked_rank": reranked_rank,
                "baseline_top_candidate_index": baseline_top["candidate_index"],
                "reranked_top_candidate_index": reranked_top["candidate_index"],
                "baseline_top_score": float(baseline_top["retrieval_score"]),
                "reranker_top_score": float(reranked_top["reranker_score"]),
                "reranker_margin": float(reranker_margin),
                "baseline_top_is_correct": bool(
                    is_mapped and baseline_top["candidate_index"] == target
                ),
                "reranked_top_is_correct": bool(
                    is_mapped and reranked_top["candidate_index"] == target
                ),
            }
        )

    per_query = pd.DataFrame(rows)

    n_validation = int(len(per_query))
    n_mapped = int(per_query["is_mapped"].sum()) if n_validation else 0
    n_novel = n_validation - n_mapped

    summary: dict[str, float | int] = {
        "validation_year": int(config.validation_year),
        "n_reference_history_rows": int(len(reference_history)),
        "n_train_reference_rows": int(len(training_reference)),
        "n_synthetic_training_queries": int(len(synthetic_training)),
        "n_validation_queries": n_validation,
        "n_mapped_queries": n_mapped,
        "n_novel_queries": n_novel,
        "mapped_coverage": float(n_mapped / n_validation) if n_validation else 0.0,
    }
    summary.update(
        _mapped_metrics(
            per_query,
            top_ks=config.retrieval_config.top_ks,
        )
    )

    if config.acceptance_margin is not None and n_validation:
        summary.update(
            _acceptance_metrics(
                per_query,
                config.acceptance_margin,
            )
        )

    return TemporalValidationResult(
        summary=summary,
        per_query=per_query,
        coefficients=reranker.coefficients,
        training_reference_indexes=tuple(training_reference.index.tolist()),
    )


__all__ = [
    "TemporalValidationConfig",
    "TemporalValidationResult",
    "evaluate_temporal_holdout",
    "fit_temporal_reranker",
]
