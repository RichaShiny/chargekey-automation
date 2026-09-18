from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .models import train_attribute_panel


@dataclass(frozen=True)
class MatchResult:
    reference_index: int
    score: float
    method: str
    needs_review: bool
    similarities: np.ndarray | None = None


def _semantic_match(
    description: str,
    ref_descriptions: list[str],
    ref_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    fuzzy_threshold: int,
    similarity_threshold: float,
) -> MatchResult:
    fuzzy_scores = np.asarray(
        [fuzz.token_sort_ratio(description.upper(), candidate) for candidate in ref_descriptions],
        dtype=float,
    )
    best_fuzzy = int(fuzzy_scores.argmax())
    if fuzzy_scores[best_fuzzy] >= fuzzy_threshold:
        return MatchResult(best_fuzzy, fuzzy_scores[best_fuzzy] / 100.0, "fuzzy", False)

    similarities = ref_embeddings @ query_embedding
    best = int(similarities.argmax())
    score = float(similarities[best])
    return MatchResult(best, score, "embedding", score < similarity_threshold, similarities)


def match_jail_charge(
    code: str,
    description: str,
    reference: pd.DataFrame,
    ref_descriptions: list[str],
    ref_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    similarity_threshold: float,
    fuzzy_threshold: int,
) -> MatchResult:
    # Jail exact match requires normalized code AND description.
    exact_mask = (
        reference["chrg_code"].astype(str).str.upper().eq(str(code).upper())
        & pd.Series(ref_descriptions, index=reference.index).eq(str(description).upper())
    )
    exact_indexes = np.flatnonzero(exact_mask.to_numpy())
    if exact_indexes.size:
        return MatchResult(int(exact_indexes[0]), 1.0, "exact", False)
    return _semantic_match(
        description,
        ref_descriptions,
        ref_embeddings,
        query_embedding,
        fuzzy_threshold,
        similarity_threshold,
    )


def match_court_charge(
    description: str,
    reference: pd.DataFrame,
    ref_descriptions: list[str],
    ref_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    similarity_threshold: float,
    fuzzy_threshold: int,
) -> MatchResult:
    del reference  # signature symmetry; court exact matching intentionally ignores charge code.
    exact_indexes = np.flatnonzero(
        pd.Series(ref_descriptions).eq(str(description).upper()).to_numpy()
    )
    if exact_indexes.size:
        return MatchResult(int(exact_indexes[0]), 1.0, "exact", False)
    return _semantic_match(
        description,
        ref_descriptions,
        ref_embeddings,
        query_embedding,
        fuzzy_threshold,
        similarity_threshold,
    )


# Backward-compatible function name for the jail workflow.
hybrid_match = match_jail_charge


def derive_description(code: str, description_map: pd.Series) -> tuple[str, bool]:
    code = str(code).strip().upper()
    attempt = re.fullmatch(r"664/(.+)", code)
    if attempt and attempt.group(1) in description_map.index:
        return f"ATT:{description_map[attempt.group(1)]}", True

    candidates = (
        re.sub(r"\([A-Z0-9]+\)$", "", code),
        re.sub(r"[A-Z]+$", "", code),
        code.split(".")[0],
        re.sub(r"\s+(MPOL|VC|PC)/?.*$", "", code),
    )
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate != code and candidate in description_map.index:
            return str(description_map[candidate]), True
    return "", False


def predict_attribute_profile(
    embedding: np.ndarray,
    panel: dict[str, dict[str, object]],
) -> dict[str, float]:
    profile: dict[str, float] = {}
    row = embedding.reshape(1, -1)
    for column, models in panel.items():
        probabilities: list[float] = []
        for classifier in models.values():
            try:
                proba = classifier.predict_proba(row)
                value = proba[0, 1] if proba.shape[1] == 2 else float(classifier.predict(row)[0])
            except Exception:
                value = float(classifier.predict(row)[0])
            probabilities.append(float(value))
        if probabilities:
            profile[column] = float(np.mean(probabilities))
    return profile


def attribute_agreement(candidate: pd.Series, profile: dict[str, float]) -> float:
    scores: list[float] = []
    for column, probability in profile.items():
        raw = candidate.get(column)
        value = 0 if pd.isna(raw) else int(raw)
        scores.append(probability if value == 1 else 1.0 - probability)
    return float(np.mean(scores)) if scores else 0.0


def rerank_candidate(
    similarities: np.ndarray,
    query_embedding: np.ndarray,
    reference: pd.DataFrame,
    panel: dict[str, dict[str, object]],
    top_k: int,
    similarity_weight: float,
    attribute_weight: float,
) -> tuple[int, float, float]:
    profile = predict_attribute_profile(query_embedding, panel)
    indexes = similarities.argsort()[-top_k:][::-1]
    scored: list[tuple[float, int]] = []
    for index in indexes:
        agreement = attribute_agreement(reference.iloc[int(index)], profile)
        combined = similarity_weight * float(similarities[index]) + attribute_weight * agreement
        scored.append((combined, int(index)))
    scored.sort(reverse=True)
    best_score, best_index = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    return best_index, float(best_score), float(best_score - second)


__all__ = [
    "MatchResult",
    "attribute_agreement",
    "derive_description",
    "hybrid_match",
    "match_court_charge",
    "match_jail_charge",
    "predict_attribute_profile",
    "rerank_candidate",
    "train_attribute_panel",
]
