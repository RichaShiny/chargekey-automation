import pandas as pd
import pytest

from charge_key_automation.synthetic import (
    SyntheticAugmentationConfig,
    generate_synthetic_training_data,
    strip_code_formatting,
)


def _reference():
    return pd.DataFrame(
        {
            "chrg_code": ["14-33", "90.95", "18/2"],
            "chrg_desc": [
                "AGGRAVATED ASSAULT WITH DEADLY WEAPON",
                "POSSESSION OF CONTROLLED SUBSTANCE",
                "DOMESTIC VIOLENCE ASSAULT",
            ],
            "source_year": [2024, 2024, 2025],
            "chrg_sev": ["F", "F", "M"],
            "chrg_ibr_code": ["13A", "35A", "13B"],
            "chrg_type_violent": [1, 0, 1],
        },
        index=[101, 202, 303],
    )


def test_synthetic_generation_is_reproducible():
    reference = _reference()
    config = SyntheticAugmentationConfig(variants_per_row=4, seed=17)

    left = generate_synthetic_training_data(reference, config=config)
    right = generate_synthetic_training_data(reference, config=config)

    pd.testing.assert_frame_equal(left, right)


def test_validation_year_is_excluded_from_synthetic_training_data():
    synthetic = generate_synthetic_training_data(
        _reference(),
        exclude_years=[2025],
        config=SyntheticAugmentationConfig(variants_per_row=3, seed=42),
    )

    assert len(synthetic) == 6
    assert set(synthetic["reference_index"]) == {101, 202}
    assert 303 not in set(synthetic["reference_index"])


def test_every_synthetic_example_retains_its_reference_row_ground_truth():
    reference = _reference()
    synthetic = generate_synthetic_training_data(
        reference,
        config=SyntheticAugmentationConfig(variants_per_row=4, seed=42),
    )

    for row in synthetic.itertuples(index=False):
        source = reference.loc[row.reference_index]
        assert row.reference_code == source["chrg_code"]
        assert row.reference_desc == source["chrg_desc"]
        assert row.synthetic_desc
        assert row.is_synthetic is True


def test_code_perturbations_preserve_underlying_statute_characters():
    synthetic = generate_synthetic_training_data(
        _reference().iloc[:2],
        config=SyntheticAugmentationConfig(variants_per_row=4, seed=42),
    )

    for row in synthetic.itertuples(index=False):
        assert strip_code_formatting(row.synthetic_code) == strip_code_formatting(
            row.reference_code
        )


def test_synthetic_output_does_not_copy_or_invent_legal_attributes():
    synthetic = generate_synthetic_training_data(
        _reference(),
        config=SyntheticAugmentationConfig(variants_per_row=2),
    )

    forbidden = {"chrg_sev", "chrg_ibr_code", "chrg_type_violent"}
    assert forbidden.isdisjoint(synthetic.columns)


def test_excluding_years_requires_year_column():
    reference = _reference().drop(columns=["source_year"])

    with pytest.raises(ValueError, match="year column"):
        generate_synthetic_training_data(reference, exclude_years=[2025])


def test_variants_per_row_must_be_positive():
    with pytest.raises(ValueError, match="variants_per_row"):
        SyntheticAugmentationConfig(variants_per_row=0)


def test_missing_values_stay_blank_instead_of_becoming_nan_text():
    reference = pd.DataFrame(
        {
            "chrg_code": [None],
            "chrg_desc": ["ASSAULT"],
            "source_year": [2024],
        },
        index=[9],
    )

    synthetic = generate_synthetic_training_data(
        reference,
        config=SyntheticAugmentationConfig(variants_per_row=1),
    )

    assert synthetic.loc[0, "reference_code"] == ""
    assert synthetic.loc[0, "synthetic_code"] == ""


def test_whitespace_noise_is_collapsed_before_augmentation():
    reference = pd.DataFrame(
        {
            "chrg_code": ["14-33"],
            "chrg_desc": ["AGGRAVATED   ASSAULT\tWITH   DEADLY WEAPON"],
            "source_year": [2024],
        },
        index=[77],
    )

    synthetic = generate_synthetic_training_data(
        reference,
        config=SyntheticAugmentationConfig(variants_per_row=1, seed=42),
    )

    assert synthetic.loc[0, "reference_desc"] == "AGGRAVATED ASSAULT WITH DEADLY WEAPON"
