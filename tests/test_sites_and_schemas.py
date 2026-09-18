import pandas as pd

from charge_key_automation.schemas import audit_jail_schema, derive_jail_attribute_columns
from charge_key_automation.sites import COURT_SITES, JAIL_SITES, filename_prefixes


def test_project_site_sets_match_source_workflows():
    assert len(JAIL_SITES) == 16
    assert COURT_SITES == (
        "Allegheny",
        "Charleston",
        "Lucas",
        "Milwaukee",
        "New Orleans",
        "Palm Beach",
        "San Francisco",
        "Spokane",
    )


def test_dynamic_schema_preserves_site_specific_columns_without_inventing_others():
    frame = pd.DataFrame(
        columns=[
            "chrg_code",
            "chrg_desc",
            "chrg_ibr_code",
            "chrg_type_violent",
            "chrg_type_site_specific",
        ]
    )
    columns = derive_jail_attribute_columns(frame)
    assert columns == [
        "chrg_ibr_code",
        "chrg_type_violent",
        "chrg_type_site_specific",
    ]
    audit = audit_jail_schema(frame)
    assert "chrg_type_site_specific" in audit.extra_vs_canonical
    assert "chrg_type_dui" in audit.missing_from_canonical


def test_historical_filename_aliases_are_preserved():
    assert "SanFrancisco" in filename_prefixes("San Francisco")
    assert "Palm_Beach" in filename_prefixes("Palm Beach")
    assert "St_Louis" in filename_prefixes("St. Louis")
