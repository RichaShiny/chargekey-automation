# Retrieval evaluation and hard-negative mining

This stage measures candidate retrieval before synthetic examples are connected
to the production matcher.

The benchmark asks two separate questions:

1. Does the correct historical reference row appear near the top of the
   candidate list for a noisy charge?
2. Which incorrect rows are most confusable with that known positive?

Those are retrieval questions, not final classification claims.

## Retrieval score

Each query/reference pair is scored with four transparent components:

- normalized statute-code similarity;
- RapidFuzz token-sort similarity;
- RapidFuzz token-set similarity;
- character n-gram TF-IDF similarity.

The default weighted score is:

```text
0.20 * code similarity
+ 0.25 * token-sort similarity
+ 0.25 * token-set similarity
+ 0.30 * character TF-IDF similarity
```

If the query has no usable charge code, the description components are
renormalized rather than treating a missing code as evidence against a
candidate.

This scorer is a benchmark/candidate generator. It does not replace the
production SentenceTransformer + ensemble path in this change.

## Metrics

`evaluate_candidate_retrieval` reports:

- Recall@1
- Recall@3
- Recall@5
- mean reciprocal rank (MRR)
- mean rank of the correct historical row

The synthetic query's `reference_index` is the ground-truth target.

High Recall@5 means the retrieval stage usually places the correct row inside a
small candidate set. It does **not** mean the final system is correct at the
same rate. Final acceptance still needs a reranker and calibrated abstention.

## Hard negatives

For every query, the benchmark also stores the highest-scoring wrong rows.
These are hard negatives: charges that look similar enough to confuse the
retriever but are known not to be the source row.

Examples include neighboring offenses such as:

- aggravated assault vs aggravated assault with firearm;
- possession vs distribution;
- simple assault vs domestic-violence assault.

These rows are useful for training and evaluating the next pairwise reranker
because they are more informative than random negatives.

## Leakage-free temporal protocol

For a 2025 validation experiment:

1. build the reference/training pool from real 2021-2024 rows;
2. generate synthetic queries only from those 2021-2024 rows;
3. mine hard negatives only against the allowed training/reference pool;
4. evaluate final model behavior separately on real 2025 observations;
5. after model selection, refit the final 2026 system on real 2021-2025 data.

Synthetic validation examples measure controlled robustness. Real 2025 remains
the out-of-time benchmark and should not be replaced by synthetic metrics.

## Example

```python
from charge_key_automation import (
    RetrievalEvaluationConfig,
    evaluate_candidate_retrieval,
    generate_synthetic_training_data,
)

synthetic = generate_synthetic_training_data(
    historical_reference,
    exclude_years=[2025],
)

evaluation = evaluate_candidate_retrieval(
    synthetic,
    historical_reference[historical_reference["source_year"] != 2025],
    config=RetrievalEvaluationConfig(
        top_ks=(1, 3, 5),
        hard_negatives_per_query=5,
    ),
)

print(evaluation.summary)
print(evaluation.hard_negatives.head())
```
