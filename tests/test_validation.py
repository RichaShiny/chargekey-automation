from pathlib import Path

import pandas as pd
import pytest

from charge_key_automation.config import CourtPipelineConfig, JailPipelineConfig
from charge_key_automation.validation import (
    InputValidationError,
    validate_court_inputs,
    validate_jail_inputs,
)


def _write_reference(folder: Path, year: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "chrg_code": ["T1", "A1"],
            "chrg_desc": ["THEFT", "ASSAULT"],
            "chrg_type_violent": [0, 1],
        }
    ).to_csv(folder / f"SanFrancisco_{year}.csv", index=False)


def test_jail_preflight_reports_missing_data_directory(tmp_path: Path):
    report = validate_jail_inputs(
        tmp_path / "missing",
        tmp_path / "output",
        JailPipelineConfig(site_name="San Francisco", target_year="2026"),
    )
    assert not report.ok
    assert {issue.code for issue in report.errors} == {"data_dir_missing"}
    with pytest.raises(InputValidationError, match="Data directory does not exist"):
        report.raise_for_errors()


def test_jail_preflight_accepts_valid_layout(tmp_path: Path):
    data = tmp_path / "data"
    output = tmp_path / "output"
    for year in range(2021, 2026):
        _write_reference(data / f"{year}_c", year)

    diagnostic = data / "2026_d"
    diagnostic.mkdir(parents=True)
    pd.DataFrame(
        {
            "chrg_code": ["T1"],
            "chrg_desc": ["THEFT"],
            "charge_key": [0],
        }
    ).to_excel(diagnostic / "SanFrancisco_2026.xlsx", sheet_name="Sheet1", index=False)

    report = validate_jail_inputs(
        data,
        output,
        JailPipelineConfig(site_name="San Francisco", target_year="2026"),
    )
    assert report.ok
    assert not report.errors


def test_jail_preflight_catches_malformed_diagnostic(tmp_path: Path):
    data = tmp_path / "data"
    output = tmp_path / "output"
    for year in range(2021, 2026):
        _write_reference(data / f"{year}_c", year)

    diagnostic = data / "2026_d"
    diagnostic.mkdir(parents=True)
    pd.DataFrame({"chrg_code": ["T1"], "chrg_desc": ["THEFT"]}).to_excel(
        diagnostic / "SanFrancisco_2026.xlsx", sheet_name="Sheet1", index=False
    )

    report = validate_jail_inputs(
        data,
        output,
        JailPipelineConfig(site_name="San Francisco", target_year="2026"),
    )
    assert "missing_columns" in {issue.code for issue in report.errors}


def test_court_preflight_requires_finalized_or_raw_jail_key(tmp_path: Path):
    data = tmp_path / "data"
    output = tmp_path / "output"
    output.mkdir()
    for year in range(2021, 2025):
        _write_reference(data / f"{year}_c", year)

    diagnostic = data / "2026_court_d"
    diagnostic.mkdir(parents=True)
    pd.DataFrame(
        {
            "chrg_code": ["COURT-1"],
            "chrg_desc": ["THEFT"],
            "charge_key": [0],
        }
    ).to_excel(diagnostic / "SanFrancisco_2026.xlsx", sheet_name="Sheet1", index=False)

    report = validate_court_inputs(
        data,
        output,
        CourtPipelineConfig(site_name="San Francisco", target_year="2026"),
    )
    assert "finalized_jail_key_missing" in {issue.code for issue in report.errors}
