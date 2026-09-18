from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CourtPipelineConfig
from ..embedding import Embedder, SentenceTransformerEmbedder
from ..io import (
    load_court_diagnostic,
    load_finalized_jail_key,
    load_reference_frames,
    preprocess_site,
)
from ..matching import match_court_charge
from ..models import positive_class_probability, train_attribute_panel
from ..output import write_court_workbook
from ..schemas import derive_court_columns


@dataclass(frozen=True)
class CourtPipelineSummary:
    site: str
    output_path: Path
    total_new_charges: int
    confident: int
    ensemble_resolved: int
    needs_review: int
    automation_rate: float


def _normalize_embeddings(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values):
        return values
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def _panel_predictions(
    embeddings: np.ndarray,
    panel: dict[str, dict[str, object]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    votes: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    for column, models in panel.items():
        column_votes = np.zeros((len(embeddings), len(models)), dtype=int)
        column_probabilities = np.zeros((len(embeddings), len(models)), dtype=float)
        for model_index, classifier in enumerate(models.values()):
            prediction = np.asarray(classifier.predict(embeddings), dtype=int)
            column_votes[:, model_index] = prediction
            column_probabilities[:, model_index] = positive_class_probability(
                classifier, embeddings
            )
        votes[column] = column_votes
        probabilities[column] = column_probabilities
    return votes, probabilities


def _resolve_with_consensus(
    row_index: int,
    hybrid_attributes: dict[str, object],
    votes: dict[str, np.ndarray],
    probabilities: dict[str, np.ndarray],
    agreement_threshold: float,
    probability_threshold: float,
) -> tuple[dict[str, int], int, int, bool, float]:
    final = dict(hybrid_attributes)
    disagreements = 0
    resolved = 0
    needs_review = False
    minimum_confidence = 1.0

    for column in votes:
        raw = hybrid_attributes.get(column, 0)
        hybrid_value = 0 if pd.isna(raw) else int(raw)
        model_votes = votes[column][row_index]
        positive_probabilities = probabilities[column][row_index]
        majority = int(round(float(model_votes.mean())))
        agreement = float((model_votes == majority).mean())
        mean_probability = float(
            positive_probabilities.mean()
            if majority == 1
            else 1.0 - positive_probabilities.mean()
        )

        if majority == hybrid_value:
            continue
        disagreements += 1
        minimum_confidence = min(minimum_confidence, mean_probability)
        if agreement >= agreement_threshold and mean_probability >= probability_threshold:
            final[column] = majority
            resolved += 1
        else:
            needs_review = True

    return final, disagreements, resolved, needs_review, round(minimum_confidence, 4)


def run_court_pipeline(
    data_dir: str | Path,
    output_dir: str | Path,
    config: CourtPipelineConfig,
    embedder: Embedder | None = None,
) -> CourtPipelineSummary:
    config.validate()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    reference = load_finalized_jail_key(
        data_dir, output_dir, config.site_name, config.target_year
    ).drop_duplicates(["chrg_code", "chrg_desc"], keep="last").reset_index(drop=True)
    reference = reference[reference["chrg_desc"].str.len().ge(3)].reset_index(drop=True)
    if reference.empty:
        raise RuntimeError(f"Finalized jail key for {config.site_name} has no usable descriptions")

    court = load_court_diagnostic(data_dir, config.site_name, config.target_year)
    charge_key_flag = pd.to_numeric(court["charge_key"], errors="coerce").fillna(0)
    new_charges = court[charge_key_flag.eq(0)].drop_duplicates(
        ["chrg_code", "chrg_desc"]
    ).reset_index(drop=True)
    new_charges = new_charges[new_charges["chrg_desc"].str.len().ge(3)].reset_index(drop=True)
    if new_charges.empty:
        raise RuntimeError(f"No new court charges found for {config.site_name}")

    train_frames = load_reference_frames(
        data_dir, config.site_name, config.resolved_reference_years()
    )
    if not train_frames:
        raise FileNotFoundError(f"No court training keys found for {config.site_name}")
    training = preprocess_site(pd.concat(train_frames, ignore_index=True), config.site_name)
    training = training.drop_duplicates(["chrg_code", "chrg_desc"], keep="last")
    training = training[training["chrg_desc"].str.len().ge(3)].reset_index(drop=True)

    embedder = embedder or SentenceTransformerEmbedder(config.embedding_model)
    reference_descriptions = reference["chrg_desc"].astype(str).str.upper().tolist()
    court_descriptions = new_charges["chrg_desc"].astype(str).str.upper().tolist()
    training_descriptions = training["chrg_desc"].astype(str).str.upper().tolist()

    reference_embeddings = _normalize_embeddings(embedder.encode(reference_descriptions))
    court_embeddings = _normalize_embeddings(embedder.encode(court_descriptions))
    training_embeddings = _normalize_embeddings(embedder.encode(training_descriptions))

    attribute_columns, carry_columns = derive_court_columns(reference)
    train_attributes = [column for column in attribute_columns if column in training.columns]
    panel = train_attribute_panel(
        training, training_embeddings, train_attributes, config.min_class_count
    )

    rows: list[dict[str, object]] = []
    for index, charge in new_charges.iterrows():
        match = match_court_charge(
            charge["chrg_desc"],
            reference,
            reference_descriptions,
            reference_embeddings,
            court_embeddings[index],
            config.similarity_threshold,
            config.fuzzy_threshold,
        )
        candidate = reference.iloc[match.reference_index]
        row: dict[str, object] = {
            "site": config.site_name,
            "court_chrg_code": charge["chrg_code"],
            "court_chrg_desc": charge["chrg_desc"],
            "matched_jail_chrg_code": candidate["chrg_code"],
            "matched_jail_chrg_desc": candidate["chrg_desc"],
            "similarity_score": round(match.score, 4),
            "match_method": match.method,
            "low_similarity": match.needs_review,
        }
        for column in attribute_columns + carry_columns:
            row[column] = candidate.get(column)
        rows.append(row)

    results = pd.DataFrame(rows)
    votes, probabilities = _panel_predictions(court_embeddings, panel)

    resolutions = []
    for index in range(len(results)):
        hybrid_attributes = {column: results.iloc[index].get(column, 0) for column in votes}
        resolutions.append(
            _resolve_with_consensus(
                index,
                hybrid_attributes,
                votes,
                probabilities,
                config.agreement_threshold,
                config.probability_threshold,
            )
        )

    results["disagreement_count"] = [resolution[1] for resolution in resolutions]
    results["auto_resolved_count"] = [resolution[2] for resolution in resolutions]
    results["consensus_confidence"] = [resolution[4] for resolution in resolutions]
    results["needs_human_review"] = [
        resolution[3] or bool(low_similarity)
        for resolution, low_similarity in zip(resolutions, results["low_similarity"])
    ]
    for index, (final_attributes, *_rest) in enumerate(resolutions):
        for column, value in final_attributes.items():
            results.at[index, column] = value

    path = write_court_workbook(
        output_dir, config.site_name, config.target_year, results
    )

    confident = int(
        (results["disagreement_count"].eq(0) & ~results["needs_human_review"].astype(bool)).sum()
    )
    resolved = int(
        (results["disagreement_count"].gt(0) & ~results["needs_human_review"].astype(bool)).sum()
    )
    review = int(results["needs_human_review"].astype(bool).sum())
    total = len(results)
    rate = round((total - review) / total * 100, 1) if total else 0.0

    return CourtPipelineSummary(
        site=config.site_name,
        output_path=path,
        total_new_charges=total,
        confident=confident,
        ensemble_resolved=resolved,
        needs_review=review,
        automation_rate=rate,
    )
