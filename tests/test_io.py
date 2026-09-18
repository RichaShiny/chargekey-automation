import pandas as pd

from charge_key_automation.io import normalize_code, preprocess_site


def test_normalize_code_handles_missing_and_spaces():
    assert normalize_code("  pc 187 ") == "PC 187"
    assert normalize_code(None) == ""
    assert normalize_code("nan") == ""


def test_preprocess_respects_spokane_description_case():
    frame = pd.DataFrame({"chrg_code": [" a1 "], "chrg_desc": [" Theft "]})
    out = preprocess_site(frame, "Spokane")
    assert out.loc[0, "chrg_code"] == "A1"
    assert out.loc[0, "chrg_desc"] == "theft"


def test_site_file_aliases_find_san_francisco(tmp_path):
    from charge_key_automation.io import find_site_files

    path = tmp_path / "SanFrancisco_charge_key.csv"
    path.write_text("chrg_code,chrg_desc\nA,THEFT\n", encoding="utf-8")
    assert find_site_files(tmp_path, "San Francisco") == [path]
