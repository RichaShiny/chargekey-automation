"""Pairwise hard-negative reranking for charge-key candidates.

The reranker learns preferences between a known positive historical row and
high-scoring wrong rows. Training uses feature differences rather than row IDs,
so legal attributes are never synthesized and candidate identity is not a
model feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from .retrieval_eval import RetrievalEvaluationConfig, rank_query_candidates


PAIR_FEATURES = (
    "retrieval_score",
    "code_similarity",
    "token_sort_similarity",
    "token_set_similarity",
    "char_tfidf_similarity",
)


@dataclass
class PairwiseReranker:
    """Linear preference model trained on positive-vs-hard-negative differences."""

    model: LogisticRegression
    feature_columns: tuple[str, ...] = PAIR_FEATURES

    def score(self, candidates: pd.DataFrame) -> np.ndarray:
        missing = set(self.feature_columns) - set(candidates.columns)
        if missing:
            raise ValueError(f"candidate table is missing features: {sorted(missing)}")
        matrix = candidates.loc[:, self.feature_columns].to_numpy(dtype=float)
        return self.model.decision_function(matrix)

    @property
    def coefficients(self) -> dict[str, float]:
        weights = self.model.coef_[0]
        return {
            feature: float(weight)
            for feature, weight in zip(self.feature_columns, weights, strict=True)
        }


@dataclass
class PairwiseRerankerEvaluation:
    """Held-out comparison of baseline retrieval and learned reranking."""

    summary: dict[str, float | int]
    per_query: pd.DataFrame
    coefficients: dict[str, float]
    train_reference_indexes: tuple[object, ...]
    test_reference_indexes: tuple[object, ...]


def _validate_queries(
    queries: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    query_code_col: str,
    query_desc_col: str,
    target_col: str,
) -> None:
    required = {query_code_col, query_desc_col, target_col}
    missing = required - set(queries.columns)
    if missing:
        raise ValueError(f"queries are missing required columns: {sorted(missing)}")
    if reference.index.has_duplicates:
        raise ValueError("reference index must be unique")
    unknown = set(queries[target_col]) - set(reference.index)
    if unknown:
        sample = list(unknown)[:5]
        raise ValueError(f"queries contain targets absent from reference: {sample}")


def build_pairwise_training_data(
    queries: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    retrieval_config: RetrievalEvaluationConfig | None = None,
    negatives_per_query: int = 5,
    query_code_col: str = "synthetic_code",
    query_desc_col: str = "synthetic_desc",
    target_col: str = "reference_index",
) -> tuple[np.ndarray, np.ndarray]:
    """Build symmetric pairwise examples from positives and hard negatives.

    For each query and each hard negative, two examples are created:

    - positive_features - negative_features -> label 1
    - negative_features - positive_features -> label 0

    This symmetry makes the linear decision boundary interpretable as a ranking
    preference and avoids learning an arbitrary global intercept.
    """

    if negatives_per_query < 1:
        raise ValueError("negatives_per_query must be at least 1")

    _validate_queries(
        queries,
        reference,
        query_code_col=query_code_col,
        query_desc_col=query_desc_col,
        target_col=target_col,
    )

    retrieval_config = retrieval_config or RetrievalEvaluationConfig()
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []

    for _, query in queries.iterrows():
        ranked = rank_query_candidates(
            query[query_code_col],
            query[query_desc_col],
            reference,
            config=retrieval_config,
        )
        target = query[target_col]
        positive = ranked[ranked["candidate_index"] == target]
        if positive.empty:
            raise RuntimeError(f"target {target!r} disappeared from candidate ranking")

        positive_vector = positive.iloc[0].loc[list(PAIR_FEATURES)].to_numpy(dtype=float)
        negatives = ranked[ranked["candidate_index"] != target].head(negatives_per_query)

        for _, negative in negatives.iterrows():
            negative_vector = negative.loc[list(PAIR_FEATURES)].to_numpy(dtype=float)
            delta = positive_vector - negative_vector
            x_rows.extend([delta, -delta])
            y_rows.extend([1, 0])

    if not x_rows:
        raise ValueError("no positive/negative training pairs could be constructed")

    return np.vstack(x_rows), np.asarray(y_rows, dtype=int)


def fit_pairwise_reranker(
    queries: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    retrieval_config: RetrievalEvaluationConfig | None = None,
    negatives_per_query: int = 5,
    regularization_c: float = 1.0,
    random_state: int = 42,
    query_code_col: str = "synthetic_code",
    query_desc_col: str = "synthetic_desc",
    target_col: str = "reference_index",
) -> PairwiseReranker:
    """Fit a linear pairwise preference model on known positives and hard negatives."""

    if regularization_c <= 0:
        raise ValueError("regularization_c must be positive")

    x_train, y_train = build_pairwise_training_data(
        queries,
        reference,
        retrieval_config=retrieval_config,
        negatives_per_query=negatives_per_query,
        query_code_col=query_code_col,
        query_desc_col=query_desc_col,
        target_col=target_col,
    )

    model = LogisticRegression(
        C=regularization_c,
        fit_intercept=False,
        solver="liblinear",
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return PairwiseReranker(model=model)


def rerank_query_candidates(
    query_code: object,
    query_desc: object,
    reference: pd.DataFrame,
    reranker: PairwiseReranker,
    *,
    retrieval_config: RetrievalEvaluationConfig | None = None,
) -> pd.DataFrame:
    """Rerank the baseline candidate table with the learned pairwise preference."""

    ranked = rank_query_candidates(
        query_code,
        query_desc,
        reference,
        config=retrieval_config,
    ).copy()

    ranked["reranker_score"] = reranker.score(ranked)

    # Preserve the baseline ordering as deterministic tie-breaker.
    original_rank = ranked["rank"].to_numpy(dtype=int)
    scores = ranked["reranker_score"].to_numpy(dtype=float)
    order = np.lexsort((original_rank, -scores))

    reranked = ranked.iloc[order].reset_index(drop=True)
    reranked["baseline_rank"] = reranked["rank"].astype(int)
    reranked["rank"] = np.arange(1, len(reranked) + 1)
    return reranked


def _metric_summary(
    per_query: pd.DataFrame,
    *,
    top_ks: tuple[int, ...],
) -> dict[str, float | int]:
    summary: dict[str, float | int] = {
        "n_test_queries": int(len(per_query)),
        "baseline_mrr": float((1.0 / per_query["baseline_rank"]).mean()),
        "reranked_mrr": float((1.0 / per_query["reranked_rank"]).mean()),
        "baseline_mean_rank": float(per_query["baseline_rank"].mean()),
        "reranked_mean_rank": float(per_query["reranked_rank"].mean()),
    }

    summary["mrr_delta"] = float(summary["reranked_mrr"] - summary["baseline_mrr"])
    summary["mean_rank_delta"] = float(
        summary["reranked_mean_rank"] - summary["baseline_mean_rank"]
    )

    for k in sorted(set(top_ks)):
        baseline = float((per_query["baseline_rank"] <= k).mean())
        reranked = float((per_query["reranked_rank"] <= k).mean())
        summary[f"baseline_recall_at_{k}"] = baseline
        summary[f"reranked_recall_at_{k}"] = reranked
        summary[f"recall_at_{k}_delta"] = reranked - baseline

    summary["improved_queries"] = int(
        (per_query["reranked_rank"] < per_query["baseline_rank"]).sum()
    )
    summary["worsened_queries"] = int(
        (per_query["reranked_rank"] > per_query["baseline_rank"]).sum()
    )
    summary["unchanged_queries"] = int(
        (per_query["reranked_rank"] == per_query["baseline_rank"]).sum()
    )
    return summary


def evaluate_pairwise_reranker(
    queries: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    retrieval_config: RetrievalEvaluationConfig | None = None,
    negatives_per_query: int = 5,
    test_size: float = 0.30,
    random_state: int = 42,
    regularization_c: float = 1.0,
    query_code_col: str = "synthetic_code",
    query_desc_col: str = "synthetic_desc",
    target_col: str = "reference_index",
) -> PairwiseRerankerEvaluation:
    """Compare baseline retrieval and reranking on held-out reference groups.

    The split is grouped by reference_index, so synthetic siblings derived from
    the same historical row can never appear in both training and evaluation.
    """

    _validate_queries(
        queries,
        reference,
        query_code_col=query_code_col,
        query_desc_col=query_desc_col,
        target_col=target_col,
    )

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    groups = queries[target_col].to_numpy()
    unique_groups = pd.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("at least two distinct reference groups are required")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_positions, test_positions = next(
        splitter.split(queries, groups=groups)
    )

    train_queries = queries.iloc[train_positions]
    test_queries = queries.iloc[test_positions]

    train_groups = set(train_queries[target_col])
    test_groups = set(test_queries[target_col])
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(f"group leakage detected across split: {sorted(overlap)}")

    retrieval_config = retrieval_config or RetrievalEvaluationConfig()

    reranker = fit_pairwise_reranker(
        train_queries,
        reference,
        retrieval_config=retrieval_config,
        negatives_per_query=negatives_per_query,
        regularization_c=regularization_c,
        random_state=random_state,
        query_code_col=query_code_col,
        query_desc_col=query_desc_col,
        target_col=target_col,
    )

    rows: list[dict[str, object]] = []

    for query_index, query in test_queries.iterrows():
        baseline = rank_query_candidates(
            query[query_code_col],
            query[query_desc_col],
            reference,
            config=retrieval_config,
        )
        reranked = rerank_query_candidates(
            query[query_code_col],
            query[query_desc_col],
            reference,
            reranker,
            retrieval_config=retrieval_config,
        )

        target = query[target_col]
        baseline_row = baseline[baseline["candidate_index"] == target]
        reranked_row = reranked[reranked["candidate_index"] == target]

        if baseline_row.empty or reranked_row.empty:
            raise RuntimeError(f"target {target!r} disappeared during evaluation")

        rows.append(
            {
                "query_index": query_index,
                "reference_index": target,
                "baseline_rank": int(baseline_row.iloc[0]["rank"]),
                "reranked_rank": int(reranked_row.iloc[0]["rank"]),
                "baseline_score": float(baseline_row.iloc[0]["retrieval_score"]),
                "reranker_score": float(reranked_row.iloc[0]["reranker_score"]),
            }
        )

    per_query = pd.DataFrame(rows)
    summary = _metric_summary(per_query, top_ks=retrieval_config.top_ks)
    summary["n_train_queries"] = int(len(train_queries))
    summary["n_train_reference_groups"] = int(len(train_groups))
    summary["n_test_reference_groups"] = int(len(test_groups))

    return PairwiseRerankerEvaluation(
        summary=summary,
        per_query=per_query,
        coefficients=reranker.coefficients,
        train_reference_indexes=tuple(sorted(train_groups, key=str)),
        test_reference_indexes=tuple(sorted(test_groups, key=str)),
    )


__all__ = [
    "PAIR_FEATURES",
    "PairwiseReranker",
    "PairwiseRerankerEvaluation",
    "build_pairwise_training_data",
    "evaluate_pairwise_reranker",
    "fit_pairwise_reranker",
    "rerank_query_candidates",
]
