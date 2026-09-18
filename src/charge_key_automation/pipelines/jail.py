from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import JailPipelineConfig
from ..embedding import Embedder, SentenceTransformerEmbedder
from ..io import (
    load_current_key,
    load_jail_diagnostic,
    load_reference_frames,
    normalize_code,
    prepare_reference,
)
from ..matching import (
    derive_description,
    hybrid_match,
    rerank_candidate,
    train_attribute_panel,
)
from ..output import write_jail_intermediary_workbook


@dataclass(frozen=True)
class JailPipelineSummary:
    site: str
    output_path: Path
    total_new_charges: int
    auto_classified: int
    needs_review: int
    automation_rate: float


def _description_maps(frames: list[pd.DataFrame], ref: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    newest = ref[ref["chrg_desc"].str.len().ge(3)].copy()
    code_to_desc = (
        newest.drop_duplicates("chrg_code", keep="last")
        .set_index("chrg_code")["chrg_desc"]
        .to_dict()
    )

    all_ref = pd.concat(frames, ignore_index=True).copy()
    all_ref["chrg_code"] = all_ref["chrg_code"].map(normalize_code)
    all_ref["chrg_desc"] = all_ref["chrg_desc"].fillna("").astype(str).str.strip()
    all_ref = all_ref[all_ref["chrg_desc"].str.len().ge(3)]
    code_to_desc_all = (
        all_ref.sort_values("source_year").drop_duplicates("chrg_code", keep="last")
        .set_index("chrg_code")["chrg_desc"]
        .to_dict()
    )
    return code_to_desc, code_to_desc_all


def _fill_missing_descriptions(
    new_rows: pd.DataFrame,
    code_to_desc: dict[str, str],
    code_to_desc_all: dict[str, str],
) -> pd.DataFrame:
    out = new_rows.copy()
    missing = out["chrg_desc"].str.len().lt(3)
    for idx in out.index[missing]:
        code = out.at[idx, "chrg_code"]
        out.at[idx, "chrg_desc"] = code_to_desc.get(code) or code_to_desc_all.get(code, "")
    return out


def run_jail_pipeline(
    data_dir: str | Path,
    output_dir: str | Path,
    config: JailPipelineConfig,
    embedder: Embedder | None = None,
) -> JailPipelineSummary:
    config.validate()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    reference_years = config.resolved_reference_years()

    frames = load_reference_frames(data_dir, config.site_name, reference_years)
    ref = prepare_reference(frames, config.site_name)
    attr_cols = (["chrg_ibr_code"] if "chrg_ibr_code" in ref.columns else []) + [
        c for c in ref.columns if c.startswith("chrg_type")
    ]

    previous_year = reference_years[-1]
    current_key = load_current_key(
        data_dir, output_dir, config.site_name, config.target_year, previous_year
    )
    diag = load_jail_diagnostic(data_dir, config.site_name, config.target_year)
    charge_key_flag = pd.to_numeric(diag["charge_key"], errors="coerce").fillna(0)
    raw_new = diag[charge_key_flag.eq(0)].reset_index(drop=True)
    if raw_new.empty:
        raise RuntimeError(f"No new charges found for {config.site_name}")

    code_to_desc, code_to_desc_all = _description_maps(frames, ref)
    raw_new = _fill_missing_descriptions(raw_new, code_to_desc, code_to_desc_all)

    has_desc = raw_new["chrg_desc"].str.len().ge(3) & ~raw_new["chrg_desc"].str.endswith("-")
    with_desc = raw_new[has_desc].drop_duplicates(["chrg_code", "chrg_desc"]).reset_index(drop=True)
    code_only = raw_new[~has_desc & raw_new["chrg_code"].ne("")].drop_duplicates("chrg_code").reset_index(drop=True)

    embedder = embedder or SentenceTransformerEmbedder(config.embedding_model)
    ref_descriptions = ref["chrg_desc"].astype(str).str.upper().tolist()
    ref_embeddings = embedder.encode(ref_descriptions)
    query_embeddings = embedder.encode(with_desc["chrg_desc"].astype(str).str.upper().tolist()) if not with_desc.empty else np.empty((0, ref_embeddings.shape[1]))

    model_attr_cols = [c for c in attr_cols if c.startswith("chrg_type")]
    panel = train_attribute_panel(ref, ref_embeddings, model_attr_cols, config.min_class_count)

    description_map = (
        ref[ref["chrg_desc"].str.len().ge(3)]
        .sort_values("source_year")
        .drop_duplicates("chrg_code", keep="last")
        .set_index("chrg_code")["chrg_desc"]
    )

    results: list[dict[str, object]] = []
    ambiguous: list[tuple[int, int, np.ndarray]] = []

    for query_index, row in with_desc.iterrows():
        match = hybrid_match(
            row["chrg_code"], row["chrg_desc"], ref, ref_descriptions,
            ref_embeddings, query_embeddings[query_index],
            config.similarity_threshold, config.fuzzy_threshold,
        )
        candidate = ref.iloc[match.reference_index]
        matched_desc = str(candidate.get("chrg_desc", "")).strip()
        derived = False
        if not matched_desc:
            matched_desc, derived = derive_description(str(candidate.get("chrg_code", "")), description_map)
        if not matched_desc:
            matched_desc = code_to_desc_all.get(str(candidate.get("chrg_code", "")).upper(), "")
            derived = bool(matched_desc)

        record: dict[str, object] = {
            "input_chrg_code": row["chrg_code"],
            "input_chrg_desc": row["chrg_desc"],
            "matched_chrg_code": candidate["chrg_code"],
            "matched_chrg_desc": matched_desc,
            "desc_derived": derived,
            "similarity_score": round(match.score, 4),
            "match_method": match.method,
            "needs_human_review": match.needs_review,
        }
        for col in attr_cols:
            record[col] = candidate.get(col)
        results.append(record)
        if match.needs_review and match.similarities is not None:
            ambiguous.append((len(results) - 1, query_index, match.similarities))

    ref_valid = ref[ref["chrg_code"].ne("")]
    code_groups = ref_valid.groupby("chrg_code")
    for _, row in code_only.iterrows():
        code = row["chrg_code"]
        if code in code_groups.groups:
            group = code_groups.get_group(code)
            comparable_attrs = [c for c in model_attr_cols if c in group.columns]
            conflicting = bool(
                comparable_attrs and (group[comparable_attrs].nunique(dropna=False) > 1).any()
            )
            described = group[group["chrg_desc"].str.len().gt(0)]
            candidate = (described if not described.empty else group).sort_values("source_year").iloc[-1]
            matched_desc = str(candidate.get("chrg_desc", "")).strip()
            derived = False
            if not matched_desc:
                matched_desc, derived = derive_description(code, description_map)
            record = {
                "input_chrg_code": code,
                "input_chrg_desc": row.get("chrg_desc", ""),
                "matched_chrg_code": code,
                "matched_chrg_desc": matched_desc,
                "desc_derived": derived,
                "similarity_score": 1.0,
                "match_method": "exact_code_conflict" if conflicting else "exact_code",
                "needs_human_review": conflicting,
            }
            for col in attr_cols:
                record[col] = candidate.get(col)
        else:
            matched_desc, derived = derive_description(code, description_map)
            record = {
                "input_chrg_code": code,
                "input_chrg_desc": row.get("chrg_desc", ""),
                "matched_chrg_code": None,
                "matched_chrg_desc": matched_desc,
                "desc_derived": derived,
                "similarity_score": 0.0,
                "match_method": "code_only_unmatched",
                "needs_human_review": True,
            }
            for col in attr_cols:
                record[col] = None
        results.append(record)

    results_df = pd.DataFrame(results)

    for result_index, query_index, similarities in ambiguous:
        if not panel:
            break
        best_index, combined, margin = rerank_candidate(
            similarities, query_embeddings[query_index], ref, panel,
            config.top_k, config.similarity_weight, config.attribute_weight,
        )
        if combined < config.rerank_threshold or margin < config.margin_threshold:
            continue
        candidate = ref.iloc[best_index]
        results_df.at[result_index, "matched_chrg_code"] = candidate["chrg_code"]
        results_df.at[result_index, "matched_chrg_desc"] = candidate["chrg_desc"]
        results_df.at[result_index, "similarity_score"] = round(combined, 4)
        results_df.at[result_index, "match_method"] = "ensemble"
        results_df.at[result_index, "needs_human_review"] = False
        for col in attr_cols:
            results_df.at[result_index, col] = candidate.get(col)

    review_indexes = results_df.index[results_df["needs_human_review"].astype(bool)]
    for idx in review_indexes:
        desc = str(results_df.at[idx, "matched_chrg_desc"] or "").strip()
        if desc:
            continue
        matched_code = normalize_code(results_df.at[idx, "matched_chrg_code"])
        input_code = normalize_code(results_df.at[idx, "input_chrg_code"])
        filled = code_to_desc_all.get(matched_code) or code_to_desc.get(matched_code)
        if not filled:
            filled = code_to_desc_all.get(input_code) or code_to_desc.get(input_code)
        if not filled:
            filled, _ = derive_description(input_code, description_map)
        if filled:
            results_df.at[idx, "matched_chrg_desc"] = filled
            results_df.at[idx, "desc_derived"] = True

    for idx in review_indexes:
        input_desc = str(results_df.at[idx, "input_chrg_desc"] or "").strip()
        matched_desc = str(results_df.at[idx, "matched_chrg_desc"] or "").strip()
        if not input_desc and not matched_desc:
            code = str(results_df.at[idx, "input_chrg_code"])
            results_df.at[idx, "input_chrg_desc"] = "NO DESCRIPTION IN SOURCE"
            results_df.at[idx, "matched_chrg_desc"] = f"CODE {code} - NOT IN REFERENCE WITH DESCRIPTION"
            results_df.at[idx, "desc_derived"] = True

    path = write_jail_intermediary_workbook(
        output_dir, config.site_name, config.target_year, current_key, results_df, attr_cols
    )
    total = len(results_df)
    needs_review = int(results_df["needs_human_review"].astype(bool).sum())
    auto_classified = total - needs_review
    rate = round(auto_classified / total * 100, 1) if total else 0.0
    return JailPipelineSummary(
        site=config.site_name,
        output_path=path,
        total_new_charges=total,
        auto_classified=auto_classified,
        needs_review=needs_review,
        automation_rate=rate,
    )
