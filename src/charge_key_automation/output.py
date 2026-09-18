from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schemas import (
    CHARGE_KEY_HEADER_COLUMNS,
    JAIL_MATCH_HEADER_COLUMNS,
    conform_columns,
)


def write_jail_intermediary_workbook(
    output_dir: str | Path,
    site_name: str,
    target_year: str,
    current_key: pd.DataFrame | None,
    results: pd.DataFrame,
    attribute_columns: list[str],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    match_columns = list(JAIL_MATCH_HEADER_COLUMNS) + attribute_columns
    charge_key_columns = list(CHARGE_KEY_HEADER_COLUMNS) + attribute_columns

    exported = results.rename(columns={"similarity_score": "simiarity_score"}).copy()
    exported["site"] = site_name
    if "filled_from_existing_ck" not in exported.columns:
        exported["filled_from_existing_ck"] = False

    added = exported[~exported["needs_human_review"].astype(bool)].copy()
    review = exported[exported["needs_human_review"].astype(bool)].copy()
    added = added.drop_duplicates(["input_chrg_code", "input_chrg_desc"])
    review = review.drop_duplicates(["input_chrg_code", "input_chrg_desc"])

    new_rows = added.rename(
        columns={"input_chrg_code": "chrg_code", "input_chrg_desc": "chrg_desc"}
    )
    key = current_key.copy() if current_key is not None else pd.DataFrame(columns=charge_key_columns)
    key["site"] = site_name
    key = pd.concat(
        [
            conform_columns(key, charge_key_columns),
            conform_columns(new_rows, charge_key_columns),
        ],
        ignore_index=True,
    ).drop_duplicates(["chrg_code", "chrg_desc"])

    path = output_dir / f"{site_name}_charge_key_intermediary_{target_year}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        key.to_excel(writer, sheet_name="Charge Key", index=False)
        conform_columns(added, match_columns).to_excel(writer, sheet_name="Charges added", index=False)
        conform_columns(review, match_columns).to_excel(
            writer, sheet_name="Charge to be Reviewed", index=False
        )
    return path


def write_court_workbook(
    output_dir: str | Path,
    site_name: str,
    target_year: str,
    results: pd.DataFrame,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    confident = results[
        results["disagreement_count"].eq(0) & ~results["needs_human_review"].astype(bool)
    ]
    resolved = results[
        results["disagreement_count"].gt(0) & ~results["needs_human_review"].astype(bool)
    ]
    review = results[results["needs_human_review"].astype(bool)]

    path = output_dir / f"{site_name}_court_{target_year}_matches_noLLM.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        confident.to_excel(writer, sheet_name="confident", index=False)
        resolved.to_excel(writer, sheet_name="ensemble_resolved", index=False)
        review.to_excel(writer, sheet_name="needs_review", index=False)
    return path


# Backward-compatible name for the jail pipeline.
write_intermediary_workbook = write_jail_intermediary_workbook
