"""Candidate-retrieval evaluation and hard-negative mining.

This module evaluates whether a noisy query can retrieve its known historical
reference row before any production reranker is changed. Synthetic queries keep
reference_index as ground truth; wrong high-scoring rows become hard negatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .synthetic import strip_code_formatting


@dataclass(frozen=True)
class RetrievalEvaluationConfig:
    """Weights and reporting settings for the retrieval benchmark."""

    code_weight: float = 0.20
    token_sort_weight: float = 0.25
    token_set_weight: float = 0.25
    char_tfidf_weight: float = 0.30
    top_ks: tuple[int, ...] = (1, 3, 5)
    hard_negatives_per_query: int = 5
    char_ngram_range: tuple[int, int] = (3, 5)

    def __post_init__(self) -> None:
        weights = (
            self.code_weight,
            self.token_sort_weight,
            self.token_set_weight,
            self.char_tfidf_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("retrieval weights must be non-negative")
        if sum(weights) <= 0:
            raise ValueError("at least one retrieval weight must be positive")
        if not self.top_ks or any(k < 1 for k in self.top_ks):
            raise ValueError("top_ks must contain positive integers")
        if self.hard_negatives_per_query < 1:
            raise ValueError("hard_negatives_per_query must be at least 1")
        low, high = self.char_ngram_range
        if low < 1 or high < low:
            raise ValueError("char_ngram_range must be an increasing positive range")


@dataclass
class RetrievalEvaluation:
    """Benchmark outputs."""

    summary: dict[str, float | int]
    per_query: pd.DataFrame
    hard_negatives: pd.DataFrame


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).upper().split())


def _validate(
    queries: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    query_code_col: str,
    query_desc_col: str,
    target_col: str,
    ref_code_col: str,
    ref_desc_col: str,
) -> None:
    query_required = {query_code_col, query_desc_col, target_col}
    ref_required = {ref_code_col, ref_desc_col}

    missing_query = query_required - set(queries.columns)
    missing_ref = ref_required - set(reference.columns)

    if missing_query:
        raise ValueError(f"queries are missing required columns: {sorted(missing_query)}")
    if missing_ref:
        raise ValueError(f"reference is missing required columns: {sorted(missing_ref)}")
    if reference.index.has_duplicates:
        raise ValueError("reference index must be unique because it is the retrieval target")
    if reference.empty:
        raise ValueError("reference must contain at least one row")


def _weighted_score(
    *,
    query_code: str,
    reference_codes: list[str],
    token_sort: np.ndarray,
    token_set: np.ndarray,
    char_tfidf: np.ndarray,
    config: RetrievalEvaluationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_query_code = strip_code_formatting(query_code)
    code_similarity = np.zeros(len(reference_codes), dtype=float)

    if normalized_query_code:
        code_similarity = np.asarray(
            [
                fuzz.ratio(normalized_query_code, candidate) / 100.0
                if candidate
                else 0.0
                for candidate in reference_codes
            ],
            dtype=float,
        )

    components = [
        (config.token_sort_weight, token_sort),
        (config.token_set_weight, token_set),
        (config.char_tfidf_weight, char_tfidf),
    ]

    if normalized_query_code:
        components.insert(0, (config.code_weight, code_similarity))

    denominator = sum(weight for weight, _ in components)
    combined = sum(weight * values for weight, values in components) / denominator
    return combined, code_similarity


def _rank_one_query(
    *,
    query_code: str,
    query_desc: str,
    reference: pd.DataFrame,
    ref_codes: list[str],
    ref_descs: list[str],
    char_similarity: np.ndarray,
    config: RetrievalEvaluationConfig,
) -> pd.DataFrame:
    token_sort = np.asarray(
        [fuzz.token_sort_ratio(query_desc, candidate) / 100.0 for candidate in ref_descs],
        dtype=float,
    )
    token_set = np.asarray(
        [fuzz.token_set_ratio(query_desc, candidate) / 100.0 for candidate in ref_descs],
        dtype=float,
    )

    combined, code_similarity = _weighted_score(
        query_code=query_code,
        reference_codes=ref_codes,
        token_sort=token_sort,
        token_set=token_set,
        char_tfidf=char_similarity,
        config=config,
    )

    # Stable deterministic tie-breaker: earlier reference position wins.
    positions = np.arange(len(reference))
    order = np.lexsort((positions, -combined))

    ranked = pd.DataFrame(
        {
            "candidate_index": reference.index.to_numpy()[order],
            "candidate_position": positions[order],
            "retrieval_score": combined[order],
            "code_similarity": code_similarity[order],
            "token_sort_similarity": token_sort[order],
            "token_set_similarity": token_set[order],
            "char_tfidf_similarity": char_similarity[order],
            "candidate_code": reference.iloc[order]["chrg_code"].to_numpy(),
            "candidate_desc": reference.iloc[order]["chrg_desc"].to_numpy(),
        }
    )
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def _prepare_char_retrieval(
    queries: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    query_desc_col: str,
    ref_desc_col: str,
    config: RetrievalEvaluationConfig,
):
    ref_descs = [_text(value) for value in reference[ref_desc_col]]
    query_descs = [_text(value) for value in queries[query_desc_col]]

    corpus = ref_descs if any(ref_descs) else ["EMPTY"]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=config.char_ngram_range,
        lowercase=False,
        norm="l2",
    )
    ref_matrix = vectorizer.fit_transform(corpus)

    if len(ref_descs) != ref_matrix.shape[0]:
        raise RuntimeError("reference TF-IDF matrix does not align with reference rows")

    query_matrix = vectorizer.transform(query_descs)
    return ref_descs, ref_matrix, query_matrix


def rank_query_candidates(
    query_code: object,
    query_desc: object,
    reference: pd.DataFrame,
    *,
    config: RetrievalEvaluationConfig | None = None,
) -> pd.DataFrame:
    """Rank every reference row for one query using the benchmark scorer."""

    config = config or RetrievalEvaluationConfig()
    one_query = pd.DataFrame(
        {
            "synthetic_code": [query_code],
            "synthetic_desc": [query_desc],
            "reference_index": [reference.index[0] if len(reference) else 0],
        }
    )
    _validate(
        one_query,
        reference,
        query_code_col="synthetic_code",
        query_desc_col="synthetic_desc",
        target_col="reference_index",
        ref_code_col="chrg_code",
        ref_desc_col="chrg_desc",
    )

    ref_descs, ref_matrix, query_matrix = _prepare_char_retrieval(
        one_query,
        reference,
        query_desc_col="synthetic_desc",
        ref_desc_col="chrg_desc",
        config=config,
    )
    char_similarity = linear_kernel(query_matrix[0], ref_matrix).ravel()

    return _rank_one_query(
        query_code=_text(query_code),
        query_desc=_text(query_desc),
        reference=reference,
        ref_codes=[strip_code_formatting(value) for value in reference["chrg_code"]],
        ref_descs=ref_descs,
        char_similarity=char_similarity,
        config=config,
    )


def evaluate_candidate_retrieval(
    queries: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    config: RetrievalEvaluationConfig | None = None,
    query_code_col: str = "synthetic_code",
    query_desc_col: str = "synthetic_desc",
    target_col: str = "reference_index",
    ref_code_col: str = "chrg_code",
    ref_desc_col: str = "chrg_desc",
) -> RetrievalEvaluation:
    """Evaluate top-k retrieval and mine the highest-scoring wrong candidates."""

    config = config or RetrievalEvaluationConfig()
    _validate(
        queries,
        reference,
        query_code_col=query_code_col,
        query_desc_col=query_desc_col,
        target_col=target_col,
        ref_code_col=ref_code_col,
        ref_desc_col=ref_desc_col,
    )

    if queries.empty:
        summary: dict[str, float | int] = {"n_queries": 0, "mrr": 0.0, "mean_rank": 0.0}
        for k in sorted(set(config.top_ks)):
            summary[f"recall_at_{k}"] = 0.0
        return RetrievalEvaluation(
            summary=summary,
            per_query=pd.DataFrame(),
            hard_negatives=pd.DataFrame(),
        )

    reference_positions = {label: pos for pos, label in enumerate(reference.index)}
    unknown_targets = set(queries[target_col]) - set(reference_positions)
    if unknown_targets:
        sample = list(unknown_targets)[:5]
        raise ValueError(f"queries contain targets absent from reference: {sample}")

    ref_descs, ref_matrix, query_matrix = _prepare_char_retrieval(
        queries,
        reference,
        query_desc_col=query_desc_col,
        ref_desc_col=ref_desc_col,
        config=config,
    )
    ref_codes = [strip_code_formatting(value) for value in reference[ref_code_col]]

    per_query_rows: list[dict[str, object]] = []
    hard_negative_rows: list[dict[str, object]] = []

    for query_position, (query_index, query) in enumerate(queries.iterrows()):
        char_similarity = linear_kernel(query_matrix[query_position], ref_matrix).ravel()

        ranked = _rank_one_query(
            query_code=_text(query[query_code_col]),
            query_desc=_text(query[query_desc_col]),
            reference=reference,
            ref_codes=ref_codes,
            ref_descs=ref_descs,
            char_similarity=char_similarity,
            config=config,
        )

        target = query[target_col]
        target_match = ranked[ranked["candidate_index"] == target]
        if target_match.empty:
            raise RuntimeError(f"target {target!r} disappeared from ranked candidates")

        target_row = target_match.iloc[0]
        rank = int(target_row["rank"])
        top_row = ranked.iloc[0]

        detail: dict[str, object] = {
            "query_index": query_index,
            "reference_index": target,
            "rank": rank,
            "reciprocal_rank": 1.0 / rank,
            "correct_score": float(target_row["retrieval_score"]),
            "top_candidate_index": top_row["candidate_index"],
            "top_candidate_score": float(top_row["retrieval_score"]),
            "top_is_correct": bool(top_row["candidate_index"] == target),
        }
        for k in sorted(set(config.top_ks)):
            detail[f"hit_at_{k}"] = rank <= k
        per_query_rows.append(detail)

        negatives = ranked[ranked["candidate_index"] != target].head(
            config.hard_negatives_per_query
        )
        for negative in negatives.itertuples(index=False):
            hard_negative_rows.append(
                {
                    "query_index": query_index,
                    "reference_index": target,
                    "negative_rank": int(negative.rank),
                    "negative_index": negative.candidate_index,
                    "negative_code": negative.candidate_code,
                    "negative_desc": negative.candidate_desc,
                    "retrieval_score": float(negative.retrieval_score),
                    "code_similarity": float(negative.code_similarity),
                    "token_sort_similarity": float(negative.token_sort_similarity),
                    "token_set_similarity": float(negative.token_set_similarity),
                    "char_tfidf_similarity": float(negative.char_tfidf_similarity),
                }
            )

    per_query = pd.DataFrame(per_query_rows)
    hard_negatives = pd.DataFrame(hard_negative_rows)

    summary = {
        "n_queries": int(len(per_query)),
        "mrr": float(per_query["reciprocal_rank"].mean()),
        "mean_rank": float(per_query["rank"].mean()),
    }
    for k in sorted(set(config.top_ks)):
        summary[f"recall_at_{k}"] = float(per_query[f"hit_at_{k}"].mean())

    return RetrievalEvaluation(
        summary=summary,
        per_query=per_query,
        hard_negatives=hard_negatives,
    )


def mine_hard_negatives(
    queries: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    config: RetrievalEvaluationConfig | None = None,
) -> pd.DataFrame:
    """Return only hard negatives from the retrieval benchmark."""

    return evaluate_candidate_retrieval(
        queries,
        reference,
        config=config,
    ).hard_negatives


__all__ = [
    "RetrievalEvaluation",
    "RetrievalEvaluationConfig",
    "evaluate_candidate_retrieval",
    "mine_hard_negatives",
    "rank_query_candidates",
]
