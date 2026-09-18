from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


JAIL_MATCH_HEADER_COLUMNS: tuple[str, ...] = (
    "site",
    "input_chrg_code",
    "input_chrg_desc",
    "matched_chrg_code",
    "matched_chrg_desc",
    "desc_derived",
    # Historical misspelling retained only in the exported workbook contract.
    "simiarity_score",
    "match_method",
    "needs_human_review",
    "filled_from_existing_ck",
)

CHARGE_KEY_HEADER_COLUMNS: tuple[str, ...] = ("site", "chrg_code", "chrg_desc")

COURT_RESULT_HEADER_COLUMNS: tuple[str, ...] = (
    "site",
    "court_chrg_code",
    "court_chrg_desc",
    "matched_jail_chrg_code",
    "matched_jail_chrg_desc",
    "similarity_score",
    "match_method",
    "low_similarity",
    "disagreement_count",
    "auto_resolved_count",
    "consensus_confidence",
    "needs_human_review",
)

CANONICAL_ATTRIBUTE_COLUMNS: frozenset[str] = frozenset(
    {
        "chrg_ibr_code",
        "chrg_type_violent",
        "chrg_type_pub",
        "chrg_type_dui",
        "chrg_type_traf",
        "chrg_type_theft",
        "chrg_type_oth",
        "chrg_type_ucr_homicide",
        "chrg_type_ucr_rape",
        "chrg_type_ucr_robbery",
        "chrg_type_ucr_aggassault",
        "chrg_type_ucr_burglary",
        "chrg_type_ucr_larctheft",
        "chrg_type_ucr_mvtheft",
        "chrg_type_ucr_arson",
        "chrg_type_ucr_person",
        "chrg_type_ucr_property",
        "chrg_type_fta",
        "chrg_type_ftc",
        "chrg_type_dv",
        "chrg_type_drug",
        "chrg_type_drug_use",
        "chrg_type_drug_poss_simp",
        "chrg_type_drug_poss",
        "chrg_type_drug_dist",
        "chrg_type_drug_manf",
        "chrg_type_drug_marj",
        "chrg_type_drug_para",
        "chrg_type_weapon",
        "chrg_type_weapon_use",
        "chrg_type_weapon_poss",
        "chrg_type_gun",
        "chrg_type_gun_use",
        "chrg_type_gun_poss",
        "chrg_type_explos",
        "chrg_type_explos_poss",
        "chrg_type_hold",
        "chrg_type_hold_fed",
        "chrg_type_hold_bench",
        "chrg_type_child",
        "chrg_type_child_violent",
    }
)


@dataclass(frozen=True)
class SchemaAudit:
    site_columns: tuple[str, ...]
    missing_from_canonical: tuple[str, ...]
    extra_vs_canonical: tuple[str, ...]


def derive_jail_attribute_columns(reference: pd.DataFrame) -> list[str]:
    """Preserve the schema actually present in a site's historical charge key.

    The project intentionally does not force one global attribute schema onto all
    jurisdictions. `chrg_ibr_code` is carried only when present, followed by every
    site-specific `chrg_type*` field in source-column order.
    """

    columns: list[str] = []
    if "chrg_ibr_code" in reference.columns:
        columns.append("chrg_ibr_code")
    columns.extend(c for c in reference.columns if c.startswith("chrg_type"))
    return list(dict.fromkeys(columns))


def derive_court_columns(reference: pd.DataFrame) -> tuple[list[str], list[str]]:
    attribute_columns = [c for c in reference.columns if c.startswith("chrg_type")]
    carry_columns = [c for c in ("chrg_ibr_code", "chrg_sev") if c in reference.columns]
    return attribute_columns, carry_columns


def audit_jail_schema(reference: pd.DataFrame) -> SchemaAudit:
    site_columns = tuple(derive_jail_attribute_columns(reference))
    site_set = set(site_columns)
    missing = tuple(sorted(CANONICAL_ATTRIBUTE_COLUMNS - site_set))
    extra = tuple(sorted(site_set - CANONICAL_ATTRIBUTE_COLUMNS))
    return SchemaAudit(site_columns, missing, extra)


def conform_columns(df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = None
    return out[list(columns)]
