# Synthetic charge augmentation

The synthetic layer is designed to improve matching robustness without creating
synthetic legal labels.

## Ground-truth unit

Every generated example keeps the exact historical row index that produced it:

- reference code
- reference description
- synthetic code
- synthetic description
- reference row index

The reference-row index is the supervised target. Once a downstream matcher
selects that row, legal attributes such as `chrg_sev`, `chrg_ibr_code`, and
`chrg_type_*` must be copied from the original reference row rather than
predicted or regenerated.

## Perturbations

The current generator applies controlled surface-form noise that is common in
charge descriptions and statute codes:

- legal abbreviations such as `AGGRAVATED -> AGG` and `POSSESSION -> POSS`
- punctuation variation
- whitespace normalization
- removal of low-information tokens
- a single internal-character typo
- statute separator changes such as `14-33 -> 14.33`

Code perturbations preserve the underlying alphanumeric statute characters.

## Temporal validation

Synthetic examples should never be generated from the held-out validation year.
For a 2025 temporal validation:

1. use real 2021-2024 reference rows for model fitting;
2. generate synthetic variants only from those 2021-2024 rows;
3. evaluate on real 2025 observations;
4. after evaluation, fit the final 2026 system using real 2021-2025 history.

The helper supports this directly:

~~~python
synthetic = generate_synthetic_training_data(
    historical_reference,
    exclude_years=[2025],
    config=SyntheticAugmentationConfig(
        variants_per_row=4,
        seed=42,
    ),
)
~~~

This prevents synthetic augmentation from leaking held-out 2025 charge wording
into validation.
