# Pairwise hard-negative reranking

This stage learns how to order candidate charge-key rows after retrieval. It is
still an offline benchmark layer and does not change the production matcher.

## Why pairwise ranking

The retrieval benchmark produces a correct historical row plus several
high-scoring wrong rows. Those hard negatives are more informative than random
negative examples.

For each query/negative pair, the training data contains:

```text
positive_features - negative_features -> preferred
negative_features - positive_features -> not preferred
```

The model therefore learns a preference direction between candidate pairs
instead of learning charge-row identity.

The current feature vector contains:

- baseline retrieval score
- normalized statute-code similarity
- token-sort similarity
- token-set similarity
- character n-gram TF-IDF similarity

The model is a regularized linear logistic preference model with no intercept.
Because scoring is linear, ranking candidates by the learned utility is
consistent with the pairwise difference objective.

## Leakage control

Synthetic variants derived from one historical row are grouped by
`reference_index` during train/test splitting.

That means all noisy siblings of the same source row remain entirely in the
training split or entirely in the test split.

This avoids a common synthetic-evaluation failure mode where the model trains
on one corrupted version of a charge and is evaluated on another corrupted
version of the same charge.

## Evaluation

`evaluate_pairwise_reranker` compares the baseline candidate ordering against
the learned reranker on held-out reference groups.

Reported metrics include:

- baseline and reranked Recall@1/3/5
- baseline and reranked mean reciprocal rank
- baseline and reranked mean correct-row rank
- metric deltas
- number of improved, worsened, and unchanged queries
- learned feature coefficients

A positive synthetic benchmark is evidence that the ranking objective can learn
from controlled perturbations and hard negatives. It is **not** a substitute
for real out-of-time evaluation.

## Real-year protocol

For the real experiment:

1. build historical reference/training data from 2021-2024;
2. generate synthetic variants only from 2021-2024;
3. mine hard negatives from the allowed historical reference pool;
4. tune the reranker using grouped synthetic validation only;
5. evaluate the chosen configuration once on real 2025 data;
6. report precision, recall, review rate, and ranking metrics on that real holdout;
7. only after model selection, refit the 2026 system using real 2021-2025 history.

The real 2025 result remains the decision point for whether reranking should be
enabled in production.

## Example

```python
from charge_key_automation import (
    RetrievalEvaluationConfig,
    evaluate_pairwise_reranker,
)

result = evaluate_pairwise_reranker(
    synthetic_queries,
    historical_reference,
    retrieval_config=RetrievalEvaluationConfig(top_ks=(1, 3, 5)),
    negatives_per_query=5,
    test_size=0.30,
    random_state=42,
)

print(result.summary)
print(result.coefficients)
```
