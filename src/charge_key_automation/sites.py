from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SiteRule:
    """Jurisdiction-specific source-file and normalization rules."""

    description_case: str = "upper"
    drop_missing_code: bool = False
    filename_prefixes: tuple[str, ...] = ()


JAIL_SITES: tuple[str, ...] = (
    "Allegheny",
    "Buncombe",
    "Charleston",
    "Cook",
    "Harris",
    "Lucas",
    "Mecklenburg",
    "Milwaukee",
    "Multnomah",
    "New Orleans",
    "Palm Beach",
    "Pennington",
    "Pima",
    "San Francisco",
    "Spokane",
    "St. Louis",
)

# Court pipeline source explicitly defines these eight jurisdictions.
COURT_SITES: tuple[str, ...] = (
    "Allegheny",
    "Charleston",
    "Lucas",
    "Milwaukee",
    "New Orleans",
    "Palm Beach",
    "San Francisco",
    "Spokane",
)

SITE_RULES: dict[str, SiteRule] = {
    "Allegheny": SiteRule(),
    "Buncombe": SiteRule(drop_missing_code=True),
    "Charleston": SiteRule(),
    "Cook": SiteRule(),
    "Harris": SiteRule(),
    "Lucas": SiteRule(),
    "Mecklenburg": SiteRule(),
    "Milwaukee": SiteRule(),
    "Multnomah": SiteRule(),
    "New Orleans": SiteRule(filename_prefixes=("NewOrleans", "New_Orleans")),
    "Palm Beach": SiteRule(filename_prefixes=("PalmBeach", "Palm_Beach", "Palm Beach")),
    "Pennington": SiteRule(),
    "Pima": SiteRule(),
    "San Francisco": SiteRule(
        filename_prefixes=("San Francisco", "San_Francisco", "SanFrancisco")
    ),
    "Spokane": SiteRule(description_case="lower"),
    "St. Louis": SiteRule(filename_prefixes=("StLouis", "St_Louis")),
}

# Historical normalization discovered in the source keys.
COLUMN_RENAMES: dict[str, str] = {
    "chrg_type_FTA": "chrg_type_fta",
    "chrg_type_FTC": "chrg_type_ftc",
}

# Helper/non-schema columns that should never become charge attributes.
DROP_COLUMNS: tuple[str, ...] = ("check",)


def get_site_rule(site_name: str) -> SiteRule:
    try:
        return SITE_RULES[site_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported jurisdiction: {site_name}") from exc


def filename_prefixes(site_name: str) -> tuple[str, ...]:
    rule = get_site_rule(site_name)
    if rule.filename_prefixes:
        return rule.filename_prefixes
    return (site_name, site_name.replace(" ", "_"), site_name.replace(" ", ""))
