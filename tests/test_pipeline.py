from pathlib import Path

import numpy as np
import pandas as pd

from charge_key_automation import PipelineConfig, run_pipeline


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


def test_end_to_end_writes_expected_sheets(tmp_path: Path):
    data = tmp_path / "data"
    output = tmp_path / "output"
    for year in range(2021, 2026):
        folder = data / f"{year}_c"
        folder.mkdir(parents=True)
        pd.DataFrame({
            "chrg_code": ["T1", "A1"],
            "chrg_desc": ["THEFT", "ASSAULT"],
            "chrg_type_violent": [0, 1],
        }).to_csv(folder / f"SanFrancisco_{year}.csv", index=False)

    diagnostic = data / "2026_d"
    diagnostic.mkdir(parents=True)
    pd.DataFrame({
        "chrg_code": ["T1", "A1", "X1"],
        "chrg_desc": ["THEFT", "ASSAULT", "THEFT"],
        "charge_key": [0, 0, 0],
    }).to_excel(diagnostic / "SanFrancisco_2026.xlsx", sheet_name="Sheet1", index=False)

    summary = run_pipeline(
        data,
        output,
        PipelineConfig(site_name="San Francisco", target_year="2026", min_class_count=99),
        embedder=FakeEmbedder(),
    )

    assert summary.output_path.exists()
    workbook = pd.ExcelFile(summary.output_path)
    assert workbook.sheet_names == ["Charge Key", "Charges added", "Charge to be Reviewed"]
    assert summary.total_new_charges == 3
