# Architecture

```text
Desktop app / CLI
       |
       +----------------------+----------------------+
       |                                             |
       v                                             v
Jail pipeline                                  Court pipeline
16 jurisdictions                              8 jurisdictions
       |                                             |
       +---------------- shared modules -------------+
                         |
       sites.py      jurisdiction rules and aliases
       schemas.py    dynamic schema discovery/audit
       io.py         file loading and normalization
       embedding.py  lazy SentenceTransformer adapter
       matching.py   exact/fuzzy/semantic matching
       models.py     five-model attribute panel
       output.py     workbook contracts
```

## Jail

The jail pipeline processes rows where `charge_key == 0`, reuses historical descriptions for code-only rows when possible, and applies:

1. exact code + description match;
2. RapidFuzz token-sort match;
3. semantic embedding match;
4. top-5 ensemble attribute reranking for low-similarity cases;
5. human review when score/margin criteria are not met.

The reranker combines semantic similarity with agreement between a candidate's historical attributes and the attribute profile predicted by Random Forest, HistGradientBoosting, calibrated LinearSVC, GaussianNB, and MLP models.

## Court

Court runs downstream of jail. It loads the finalized jail charge key and applies:

1. exact **description-only** match;
2. fuzzy description match;
3. semantic description match;
4. an attribute-level five-model consensus check;
5. human review for low similarity or unresolved committee disagreement.

The court committee may override an individual attribute only when its agreement and probability thresholds are both met.
