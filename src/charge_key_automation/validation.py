from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import CourtPipelineConfig, JailPipelineConfig
from .io import find_site_files, read_charge_file


class InputValidationError(ValueError):
    """Raised when pipeline inputs fail preflight validation."""


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    workflow: str
    site_name: str
    target_year: str
    issues: tuple[ValidationIssue, ...]

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if not self.errors:
            return
        details = "\n".join(f"- {issue.message}" for issue in self.errors)
        raise InputValidationError(
            f"{self.workflow.title()} preflight failed for {self.site_name}:\n{details}"
        )


def _issue(severity: str, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message)


def _validate_output_dir(output_dir: Path, issues: list[ValidationIssue]) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        issues.append(
            _issue(
                "error",
                "output_not_directory",
                f"Output path exists but is not a directory: {output_dir}",
            )
        )


def _read_and_check(
    path: Path,
    required_columns: tuple[str, ...],
    issues: list[ValidationIssue],
    source_label: str,
    *,
    sheet_name: str | int = 0,
) -> pd.DataFrame | None:
    try:
        frame = read_charge_file(path, sheet_name=sheet_name)
    except Exception as exc:
        issues.append(
            _issue(
                "error",
                "file_unreadable",
                f"Could not read {source_label} '{path.name}': {exc}",
            )
        )
        return None

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        issues.append(
            _issue(
                "error",
                "missing_columns",
                f"{source_label} '{path.name}' is missing columns: {', '.join(missing)}",
            )
        )
        return None
    return frame


def _check_new_rows(
    frame: pd.DataFrame | None,
    issues: list[ValidationIssue],
    source_label: str,
) -> None:
    if frame is None or "charge_key" not in frame.columns:
        return
    flags = pd.to_numeric(frame["charge_key"], errors="coerce")
    if not flags.eq(0).any():
        issues.append(
            _issue(
                "warning",
                "no_new_charges",
                f"{source_label} contains no rows with charge_key == 0.",
            )
        )


def validate_jail_inputs(
    data_dir: str | Path,
    output_dir: str | Path,
    config: JailPipelineConfig,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    try:
        config.validate()
    except ValueError as exc:
        issues.append(_issue("error", "invalid_config", str(exc)))
        return ValidationReport("jail", config.site_name, config.target_year, tuple(issues))

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    _validate_output_dir(output_dir, issues)

    if not data_dir.is_dir():
        issues.append(
            _issue("error", "data_dir_missing", f"Data directory does not exist: {data_dir}")
        )
        return ValidationReport("jail", config.site_name, config.target_year, tuple(issues))

    reference_files: list[Path] = []
    missing_years: list[str] = []
    for year in config.resolved_reference_years():
        hits = find_site_files(data_dir / f"{year}_c", config.site_name)
        if hits:
            reference_files.extend(hits)
        else:
            missing_years.append(year)

    if not reference_files:
        issues.append(
            _issue(
                "error",
                "reference_missing",
                f"No historical charge-key files found for {config.site_name}.",
            )
        )
    else:
        if missing_years:
            issues.append(
                _issue(
                    "warning",
                    "reference_years_missing",
                    "No site-specific reference file found for: " + ", ".join(missing_years),
                )
            )
        attribute_columns: set[str] = set()
        for path in reference_files:
            frame = _read_and_check(
                path,
                ("chrg_code", "chrg_desc"),
                issues,
                "reference charge key",
            )
            if frame is not None:
                attribute_columns.update(
                    column for column in frame.columns if column.startswith("chrg_type")
                )
        if not attribute_columns:
            issues.append(
                _issue(
                    "warning",
                    "attribute_schema_empty",
                    "Historical keys contain no chrg_type* attribute columns; ensemble "
                    "reranking will not have attribute labels.",
                )
            )

    diagnostic_hits = find_site_files(
        data_dir / f"{config.target_year}_d", config.site_name
    )
    if not diagnostic_hits:
        issues.append(
            _issue(
                "error",
                "diagnostic_missing",
                f"No jail diagnostic found for {config.site_name} in "
                f"{config.target_year}_d.",
            )
        )
    else:
        path = diagnostic_hits[0]
        diagnostic = _read_and_check(
            path,
            ("chrg_code", "chrg_desc", "charge_key"),
            issues,
            "jail diagnostic",
            sheet_name="Sheet1" if path.suffix.lower() == ".xlsx" else 0,
        )
        _check_new_rows(diagnostic, issues, "Jail diagnostic")

    previous_year = config.resolved_reference_years()[-1]
    has_starting_key = bool(
        find_site_files(data_dir / f"{config.target_year}_c", config.site_name)
        or (output_dir / f"{config.site_name}_charge_key_intermediary_{previous_year}.xlsx").exists()
        or find_site_files(data_dir / f"{previous_year}_c", config.site_name)
    )
    if not has_starting_key:
        issues.append(
            _issue(
                "warning",
                "starting_key_missing",
                "No current or previous starting charge key was found; output will contain "
                "only newly classified rows.",
            )
        )

    return ValidationReport("jail", config.site_name, config.target_year, tuple(issues))


def validate_court_inputs(
    data_dir: str | Path,
    output_dir: str | Path,
    config: CourtPipelineConfig,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    try:
        config.validate()
    except ValueError as exc:
        issues.append(_issue("error", "invalid_config", str(exc)))
        return ValidationReport("court", config.site_name, config.target_year, tuple(issues))

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    _validate_output_dir(output_dir, issues)

    if not data_dir.is_dir():
        issues.append(
            _issue("error", "data_dir_missing", f"Data directory does not exist: {data_dir}")
        )
        return ValidationReport("court", config.site_name, config.target_year, tuple(issues))

    finalized = output_dir / f"{config.site_name}_charge_key_intermediary_{config.target_year}.xlsx"
    jail_reference: pd.DataFrame | None = None
    if finalized.exists():
        try:
            jail_reference = pd.read_excel(finalized, sheet_name="Charge Key")
        except Exception as exc:
            issues.append(
                _issue(
                    "error",
                    "finalized_key_unreadable",
                    f"Could not read finalized jail key '{finalized.name}': {exc}",
                )
            )
    else:
        raw_hits = find_site_files(
            data_dir / f"{config.target_year}_c", config.site_name
        )
        if raw_hits:
            jail_reference = _read_and_check(
                raw_hits[0],
                ("chrg_code", "chrg_desc"),
                issues,
                "raw target-year jail key",
            )
        else:
            issues.append(
                _issue(
                    "error",
                    "finalized_jail_key_missing",
                    "Court processing requires the finalized jail Charge Key or a raw "
                    f"{config.target_year}_c fallback for {config.site_name}.",
                )
            )

    if jail_reference is not None:
        missing = [
            column
            for column in ("chrg_code", "chrg_desc")
            if column not in jail_reference.columns
        ]
        if missing:
            issues.append(
                _issue(
                    "error",
                    "missing_columns",
                    "Finalized jail key is missing columns: " + ", ".join(missing),
                )
            )
        if not any(column.startswith("chrg_type") for column in jail_reference.columns):
            issues.append(
                _issue(
                    "warning",
                    "attribute_schema_empty",
                    "Finalized jail key contains no chrg_type* attributes.",
                )
            )

    diagnostic_hits = find_site_files(
        data_dir / f"{config.target_year}_court_d", config.site_name
    )
    if not diagnostic_hits:
        issues.append(
            _issue(
                "error",
                "court_diagnostic_missing",
                f"No court diagnostic found for {config.site_name} in "
                f"{config.target_year}_court_d.",
            )
        )
    else:
        path = diagnostic_hits[0]
        diagnostic = _read_and_check(
            path,
            ("chrg_code", "chrg_desc", "charge_key"),
            issues,
            "court diagnostic",
            sheet_name="Sheet1" if path.suffix.lower() == ".xlsx" else 0,
        )
        _check_new_rows(diagnostic, issues, "Court diagnostic")

    training_files: list[Path] = []
    missing_years: list[str] = []
    for year in config.resolved_reference_years():
        hits = find_site_files(data_dir / f"{year}_c", config.site_name)
        if hits:
            training_files.extend(hits)
        else:
            missing_years.append(year)

    if not training_files:
        issues.append(
            _issue(
                "error",
                "training_keys_missing",
                f"No historical training charge keys found for {config.site_name}.",
            )
        )
    elif missing_years:
        issues.append(
            _issue(
                "warning",
                "training_years_missing",
                "No court training key found for: " + ", ".join(missing_years),
            )
        )

    return ValidationReport("court", config.site_name, config.target_year, tuple(issues))
