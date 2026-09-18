from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .sites import COLUMN_RENAMES, DROP_COLUMNS, filename_prefixes, get_site_rule

MISSING_TOKENS = {"", "NAN", "NA", "N/A", "NULL", "NONE"}


def normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    return "" if text in MISSING_TOKENS else text


def normalize_description(value: object, case: str = "upper") -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.upper() in MISSING_TOKENS:
        return ""
    return text.upper() if case == "upper" else text.lower()


def find_site_files(folder: str | Path, site_name: str) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    prefixes = filename_prefixes(site_name)
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".csv", ".xlsx"}
        and any(path.name.startswith(prefix) for prefix in prefixes)
    )


def read_charge_file(path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        try:
            df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp1252", low_memory=False)
    df = df.rename(columns=COLUMN_RENAMES)
    return df.drop(columns=[column for column in DROP_COLUMNS if column in df.columns])


def require_columns(df: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def preprocess_site(df: pd.DataFrame, site_name: str) -> pd.DataFrame:
    require_columns(df, ("chrg_code", "chrg_desc"), site_name)
    out = df.copy()
    rule = get_site_rule(site_name)
    # Jail app audit normalized codes for every site, including Spokane.
    out["chrg_code"] = out["chrg_code"].map(normalize_code)
    out["chrg_desc"] = out["chrg_desc"].map(
        lambda value: normalize_description(value, rule.description_case)
    )
    if rule.drop_missing_code:
        out = out[out["chrg_code"] != ""]
    return out.reset_index(drop=True)


def load_reference_frames(
    data_dir: str | Path,
    site_name: str,
    reference_years: Iterable[str],
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for year in reference_years:
        folder = Path(data_dir) / f"{year}_c"
        for path in find_site_files(folder, site_name):
            frame = read_charge_file(path)
            require_columns(frame, ("chrg_code", "chrg_desc"), str(path))
            frame["source_year"] = str(year)
            frames.append(frame)
    return frames


def prepare_reference(frames: list[pd.DataFrame], site_name: str) -> pd.DataFrame:
    if not frames:
        raise FileNotFoundError(f"No reference charge key files found for {site_name}")
    ref = preprocess_site(pd.concat(frames, ignore_index=True), site_name)
    ref = ref.sort_values("source_year").drop_duplicates(
        subset=["chrg_code", "chrg_desc"], keep="last"
    )
    # Blank descriptions must never become embedding candidates.
    valid_desc = ref["chrg_desc"].str.len().ge(3)
    return ref[valid_desc].reset_index(drop=True)


def load_current_key(
    data_dir: str | Path,
    output_dir: str | Path,
    site_name: str,
    target_year: str,
    previous_year: str,
) -> pd.DataFrame | None:
    target_hits = find_site_files(Path(data_dir) / f"{target_year}_c", site_name)
    if target_hits:
        return preprocess_site(read_charge_file(target_hits[0]), site_name)

    intermediary = Path(output_dir) / f"{site_name}_charge_key_intermediary_{previous_year}.xlsx"
    if intermediary.exists():
        return preprocess_site(pd.read_excel(intermediary, sheet_name="Charge Key"), site_name)

    previous_hits = find_site_files(Path(data_dir) / f"{previous_year}_c", site_name)
    if previous_hits:
        return preprocess_site(read_charge_file(previous_hits[0]), site_name)
    return None


def load_jail_diagnostic(data_dir: str | Path, site_name: str, target_year: str) -> pd.DataFrame:
    hits = find_site_files(Path(data_dir) / f"{target_year}_d", site_name)
    if not hits:
        raise FileNotFoundError(f"No jail diagnostic file found for {site_name} in {target_year}_d")
    path = hits[0]
    sheet = "Sheet1" if path.suffix.lower() == ".xlsx" else 0
    frame = read_charge_file(path, sheet_name=sheet)
    require_columns(frame, ("chrg_code", "chrg_desc", "charge_key"), str(path))
    return preprocess_site(frame, site_name)


def load_finalized_jail_key(
    data_dir: str | Path,
    output_dir: str | Path,
    site_name: str,
    target_year: str,
) -> pd.DataFrame:
    """Court reference: finalized jail intermediary preferred, raw target-year key fallback."""

    intermediary = Path(output_dir) / f"{site_name}_charge_key_intermediary_{target_year}.xlsx"
    if intermediary.exists():
        return preprocess_site(pd.read_excel(intermediary, sheet_name="Charge Key"), site_name)

    raw_hits = find_site_files(Path(data_dir) / f"{target_year}_c", site_name)
    if not raw_hits:
        raise FileNotFoundError(
            f"No finalized jail intermediary or raw {target_year}_c key found for {site_name}"
        )
    return preprocess_site(read_charge_file(raw_hits[0]), site_name)


def load_court_diagnostic(data_dir: str | Path, site_name: str, target_year: str) -> pd.DataFrame:
    hits = find_site_files(Path(data_dir) / f"{target_year}_court_d", site_name)
    if not hits:
        raise FileNotFoundError(
            f"No court diagnostic file found for {site_name} in {target_year}_court_d"
        )
    path = hits[0]
    sheet = "Sheet1" if path.suffix.lower() == ".xlsx" else 0
    frame = read_charge_file(path, sheet_name=sheet)
    require_columns(frame, ("chrg_code", "chrg_desc", "charge_key"), str(path))
    return preprocess_site(frame, site_name)


# Backward-compatible name used by the first clean refactor.
load_diagnostic = load_jail_diagnostic
