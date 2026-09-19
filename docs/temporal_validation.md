# Real temporal holdout validation

This stage evaluates the experimental reranker on real observations from a
future year after all training and synthetic augmentation have been restricted
to earlier years.

The intended experiment is:

- training/reference history: 2021-2024;
- real validation observations: 2025;
- final production fit, only after model selection: 2021-2025 for 2026 use.

Synthetic metrics are useful for controlled stress tests, but the real 2025
holdout is the decision point for whether reranking should be considered for
production.

## Ground truth must be independent

The validator does not infer the correct target with the same matcher it is
evaluating.

Each real validation observation must include a `reference_index` field:

- if the 2025 charge has an independently verified correct match in the
  pre-2025 reference table, store that historical row index;
- if there is no valid pre-2025 counterpart, leave `reference_index` missing;
- if a supplied target points only to a 2025-or-later row, the validator treats
  it as novel relative to the pre-2025 candidate pool.

The verified target can come from a manually reviewed mapping, an existing
curated crosswalk, or another source independent of the matcher under test.

Do not create `reference_index` by running the same retrieval/reranking
pipeline first. That would make the evaluation circular.

## Training isolation

`fit_temporal_reranker` filters the historical reference table to rows where:

```text
source_year < validation_year
```

Only those rows are used for:

- candidate retrieval during training;
- synthetic perturbation;
- hard-negative generation;
- pairwise reranker fitting.

Rows from the validation year and future years cannot enter training.

## Mapped and novel validation rows

The validator reports two populations separately.

### Mapped

A mapped validation row has a verified historical target that exists in the
pre-validation reference pool.

For mapped rows, the report includes:

- baseline and reranked top-1 accuracy;
- Recall@1/3/5;
- mean reciprocal rank;
- mean correct-row rank;
- counts of improved, worsened, and unchanged queries.

### Novel

A novel row has no valid target in the pre-validation reference pool.

Novel rows are not assigned an artificial rank because there is no correct
candidate to rank.

They remain in the validation set so an abstention policy can be tested. This
is important because forcing every new legal charge to an old row can make
top-k metrics look better while creating production errors.

## Optional review policy

A pre-selected `acceptance_margin` can be supplied.

The margin is:

```text
top reranker score - second-best reranker score
```

It is a ranking margin, not a calibrated probability.

When a margin is supplied, the holdout report also includes:

- number of auto-matched rows;
- number and rate sent to review;
- precision among auto-matched rows;
- number of novel rows incorrectly auto-accepted;
- review rate among novel rows.

Choose the margin before inspecting the real 2025 result. Tuning the margin on
the same 2025 holdout and then reporting that score would leak the holdout into
model selection.

## Example

```python
from charge_key_automation import (
    TemporalValidationConfig,
    evaluate_temporal_holdout,
)

config = TemporalValidationConfig(
    validation_year=2025,
    variants_per_row=4,
    synthetic_seed=42,
    negatives_per_query=5,
    acceptance_margin=0.25,  # example only; pre-select before holdout evaluation
)

result = evaluate_temporal_holdout(
    reference_history,
    validation_2025,
    config=config,
)

print(result.summary)
print(result.per_query.head())
print(result.coefficients)
```

The required input shapes are conceptually:

```text
reference_history
index | chrg_code | chrg_desc | source_year | ...
   10 | A1        | ...       | 2021
   20 | B2        | ...       | 2022
   30 | C3        | ...       | 2024
   40 | D4        | ...       | 2025   <- excluded from training

validation_2025
row | chrg_code | chrg_desc | source_year | reference_index
 q1 | A1        | ...       | 2025        | 10
 q2 | NEW       | ...       | 2025        | <missing>
```

The first row is evaluated against historical row 10. The second is treated as
novel and is useful for testing whether the review policy abstains instead of
forcing a false historical match.

## Interpretation

A stronger reranker should improve ranking quality on mapped real rows without
materially increasing false auto-acceptance of novel rows.

The most production-relevant quantities are therefore not just top-1 accuracy,
but the combination of:

- auto-match precision;
- review rate;
- novel auto-accept count;
- mapped coverage;
- ranking metrics on independently verified mapped rows.

No production rollout decision is encoded in this module. Thresholds and
release criteria should be chosen explicitly and documented before looking at
the final real-year holdout.
