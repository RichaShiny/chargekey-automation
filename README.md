# Charge Key Automation

A cleaned, testable implementation of the ISLG **jail and court charge-key automation** workflows. The project keeps the original no-LLM matching strategy while separating site rules, schema discovery, data loading, matching, ensemble logic, outputs, CLI, and desktop UI into a normal `src/` package.

## What the project does

### Jail workflow

For 16 jail jurisdictions, the pipeline reads historical charge keys and the target-year diagnostic report, selects rows with `charge_key == 0`, then applies:

1. exact code + description matching;
2. fuzzy description matching with RapidFuzz;
3. SentenceTransformer semantic similarity;
4. a five-model attribute ensemble that reranks the top semantic candidates;
5. human review when the candidate is still uncertain.

Jail output preserves the existing workbook contract:

- `Charge Key`
- `Charges added`
- `Charge to be Reviewed`

### Court workflow

For the eight court sites defined by the source project, court runs **after jail**. It matches court charges against the finalized jail `Charge Key`. Because court and jail codes are different, the exact court match is description-only, followed by fuzzy and semantic matching. The same five-model family then performs attribute-level consensus checks.

Court output preserves:

- `confident`
- `ensemble_resolved`
- `needs_review`

## Site-aware schemas

This repo deliberately does not force one hard-coded attribute schema onto every jurisdiction. Jail attributes are discovered from each site's historical key at runtime:

- carry `chrg_ibr_code` when present;
- carry every `chrg_type*` column in source order;
- preserve extra site-specific charge-type columns;
- do not add canonical fields the site never had.

Historical source quirks are centralized in `sites.py`, including Spokane lowercase descriptions, Buncombe missing-code handling, filename aliases, FTA/FTC renames, and removal of the helper `check` column.

See [`docs/site_schema_notes.md`](docs/site_schema_notes.md) for the exact rules.


## Synthetic robustness evaluation

The repo also includes a leakage-aware synthetic robustness layer for candidate
retrieval experiments. Historical rows can be perturbed into noisy surface
forms while retaining the exact source-row index as ground truth.

The retrieval benchmark measures Recall@1/3/5, mean reciprocal rank, and mines
the highest-scoring incorrect rows as hard negatives. A grouped pairwise
reranker benchmark then learns from positive-vs-hard-negative feature
differences and compares baseline versus reranked ranking on held-out reference
groups.

These experiments do not fabricate legal attributes and do not replace real
out-of-time validation.

See [`docs/synthetic_augmentation.md`](docs/synthetic_augmentation.md),
[`docs/retrieval_evaluation.md`](docs/retrieval_evaluation.md), and
[`docs/pairwise_reranking.md`](docs/pairwise_reranking.md).

## Repository layout

```text
charge-key-automation/
├── src/
│   └── charge_key_automation/
│       ├── pipelines/
│       │   ├── jail.py
│       │   └── court.py
│       ├── app.py
│       ├── cli.py
│       ├── config.py
│       ├── embedding.py
│       ├── io.py
│       ├── matching.py
│       ├── models.py
│       ├── output.py
│       ├── schemas.py
│       └── sites.py
├── tests/
├── docs/
│   ├── architecture.md
│   └── site_schema_notes.md
├── .github/workflows/tests.yml
├── pyproject.toml
└── README.md
```

## Expected data layout

```text
data/
├── 2021_c/
├── 2022_c/
├── 2023_c/
├── 2024_c/
├── 2025_c/
├── 2026_c/          # optional/current jail key or court fallback
├── 2026_d/          # jail diagnostic reports
└── 2026_court_d/    # court diagnostic reports
```

Source data is intentionally not committed.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Run

Desktop app:

```bash
charge-key-app
```

Jail CLI:

```bash
charge-key jail \
  --data-dir ./data \
  --output-dir ./outputs \
  --site "San Francisco" \
  --year 2026
```

Court CLI:

```bash
charge-key court \
  --data-dir ./data \
  --output-dir ./outputs \
  --site "San Francisco" \
  --year 2026
```

## Important compatibility details

The jail Excel workbook intentionally retains the historical exported field name `simiarity_score`. Internal Python code uses the corrected name `similarity_score`.

CSV reading tries UTF-8 first and falls back to CP1252 for older source keys.

## Tests

```bash
pytest
```

The test suite covers jurisdiction normalization, filename aliases, schema preservation, exact-match differences between jail and court, GUI callback regressions, and synthetic end-to-end workbook generation.

## Data privacy

No case-level jurisdiction data is included. Keep source datasets outside Git and point the app or CLI at the local data directory.
