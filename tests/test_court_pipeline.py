from pathlib import Path

import numpy as np
import pandas as pd

from charge_key_automation import CourtPipelineConfig, run_court_pipeline
from charge_key_automation.matching import match_court_charge, match_jail_charge


class FakeEmbedder:
    def encode(self, texts):
        rows = []
        for text in texts:
            text = text.upper()
            if "THEFT" in text:
                rows.append([1.0, 0.0, 0.0])
            elif "ASSAULT" in text:
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([0.0, 0.0, 1.0])
        return np.asarray(rows, dtype=float)


def test_court_exact_match_uses_description_not_charge_code():
    ref = pd.DataFrame({"chrg_code": ["JAIL-1"], "chrg_desc": ["THEFT"]})
    emb = np.asarray([[1.0, 0.0]])
    court = match_court_charge("THEFT", ref, ["THEFT"], emb, emb[0], 0.85, 90)
    jail = match_jail_charge("COURT-99", "THEFT", ref, ["THEFT"], emb, emb[0], 0.85, 101)
    assert court.method == "exact"
    assert jail.method == "embedding"


def test_court_end_to_end_writes_three_outcome_sheets(tmp_path: Path):
    data = tmp_path / "data"
    output = tmp_path / "output"
    output.mkdir(parents=True)

    # Historical jail keys used to train the court attribute panel.
    for year in range(2021, 2025):
        folder = data / f"{year}_c"
        folder.mkdir(parents=True)
        pd.DataFrame(
            {
                "chrg_code": ["J1", "J2"],
                "chrg_desc": ["THEFT", "ASSAULT"],
                "chrg_type_violent": [0, 1],
            }
        ).to_csv(folder / f"SanFrancisco_{year}.csv", index=False)

    # Court prefers the finalized jail intermediary for the same target year.
    pd.DataFrame(
        {
            "site": ["San Francisco", "San Francisco"],
            "chrg_code": ["J1", "J2"],
            "chrg_desc": ["THEFT", "ASSAULT"],
            "chrg_type_violent": [0, 1],
            "chrg_sev": ["F", "F"],
        }
    ).to_excel(
        output / "San Francisco_charge_key_intermediary_2026.xlsx",
        sheet_name="Charge Key",
        index=False,
    )

    diagnostic = data / "2026_court_d"
    diagnostic.mkdir(parents=True)
    pd.DataFrame(
        {
            "chrg_code": ["COURT-100", "COURT-200"],
            "chrg_desc": ["THEFT", "ASSAULT"],
            "charge_key": [0, 0],
        }
    ).to_excel(
        diagnostic / "SanFrancisco_2026.xlsx", sheet_name="Sheet1", index=False
    )

    summary = run_court_pipeline(
        data,
        output,
        CourtPipelineConfig(
            site_name="San Francisco",
            target_year="2026",
            min_class_count=99,
        ),
        embedder=FakeEmbedder(),
    )

    assert summary.output_path.exists()
    workbook = pd.ExcelFile(summary.output_path)
    assert workbook.sheet_names == ["confident", "ensemble_resolved", "needs_review"]
    assert summary.total_new_charges == 2
    confident = pd.read_excel(summary.output_path, sheet_name="confident")
    assert confident["matched_jail_chrg_code"].tolist() == ["J1", "J2"]
