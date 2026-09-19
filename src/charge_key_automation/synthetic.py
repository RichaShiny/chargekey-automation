"""Deterministic synthetic augmentation for charge-key matching.

The augmentation layer perturbs charge codes/descriptions while retaining the
position of the historical reference row that generated each example. It is
intended to improve retrieval/reranking robustness, not to synthesize legal
attributes. Downstream code should always copy attributes from the matched
historical row.

Synthetic examples should be generated from training/reference years only.
Held-out validation years can be excluded explicitly with exclude_years.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Iterable

import pandas as pd


ABBREVIATIONS: dict[str, str] = {
    "AGGRAVATED": "AGG",
    "ASSAULT": "ASSLT",
    "ATTEMPT": "ATT",
    "CONTROLLED": "CTRL",
    "DISTRIBUTION": "DIST",
    "DOMESTIC": "DOM",
    "DRIVING": "DRV",
    "FAILURE": "FAIL",
    "INFLUENCE": "INFL",
    "MANUFACTURE": "MANUF",
    "MOTOR": "MTR",
    "POSSESSION": "POSS",
    "SUBSTANCE": "SUBST",
    "VEHICLE": "VEH",
    "VIOLENCE": "VIOL",
    "WEAPON": "WPN",
    "WITH": "W/",
    "WITHOUT": "W/O",
}

LOW_INFORMATION_TOKENS = {
    "A",
    "AN",
    "AND",
    "BY",
    "FOR",
    "IN",
    "OF",
    "ON",
    "OR",
    "THE",
    "TO",
    "WITH",
}


@dataclass(frozen=True)
class SyntheticAugmentationConfig:
    """Configuration for deterministic charge perturbation."""

    variants_per_row: int = 4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.variants_per_row < 1:
            raise ValueError("variants_per_row must be at least 1")


def _clean_space(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _abbreviate(description: str, rng: random.Random) -> tuple[str, bool]:
    words = {
        match.group(0).upper()
        for match in re.finditer(r"[A-Za-z]+", description)
        if match.group(0).upper() in ABBREVIATIONS
        and ABBREVIATIONS[match.group(0).upper()] != match.group(0).upper()
    }
    if not words:
        return description, False

    candidates = sorted(words)
    n_replace = rng.randint(1, min(2, len(candidates)))
    chosen = rng.sample(candidates, n_replace)
    out = description
    for word in chosen:
        out = re.sub(
            rf"\b{re.escape(word)}\b",
            ABBREVIATIONS[word],
            out,
            flags=re.IGNORECASE,
        )
    return _clean_space(out), out != description


def _drop_low_information_token(description: str, rng: random.Random) -> tuple[str, bool]:
    tokens = description.split()
    candidates = [
        i
        for i, token in enumerate(tokens)
        if re.sub(r"[^A-Za-z]", "", token).upper() in LOW_INFORMATION_TOKENS
    ]
    if not candidates or len(tokens) <= 2:
        return description, False

    del tokens[rng.choice(candidates)]
    out = _clean_space(" ".join(tokens))
    return out, out != description


def _punctuation_noise(description: str, rng: random.Random) -> tuple[str, bool]:
    positions = [i for i, ch in enumerate(description) if ch in ",;:()-./"]
    if not positions:
        return description, False

    replacement = rng.choice([" ", "", " / "])
    target = rng.choice(positions)
    out = description[:target] + replacement + description[target + 1 :]
    out = _clean_space(out)
    return out, out != description


def _single_typo(description: str, rng: random.Random) -> tuple[str, bool]:
    spans = [match.span() for match in re.finditer(r"[A-Za-z]{5,}", description)]
    if not spans:
        return description, False

    start, end = rng.choice(spans)
    word = description[start:end]
    pos = rng.randint(1, len(word) - 2)

    if rng.random() < 0.5:
        mutated = word[:pos] + word[pos + 1 :]
    else:
        chars = list(word)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        mutated = "".join(chars)

    out = description[:start] + mutated + description[end:]
    return _clean_space(out), out != description


def _code_format_noise(code: str, rng: random.Random) -> tuple[str, bool]:
    code = _clean_space(code)
    separators = [i for i, ch in enumerate(code) if ch in "-./"]
    if not separators:
        return code, False

    target = rng.choice(separators)
    original = code[target]
    choices = [value for value in [" ", "-", ".", "/"] if value != original]
    replacement = rng.choice(choices)
    out = _clean_space(code[:target] + replacement + code[target + 1 :])
    return out, out != code


def perturb_charge(
    code: str,
    description: str,
    *,
    rng: random.Random,
    variant_number: int,
) -> tuple[str, str, tuple[str, ...]]:
    """Create one controlled noisy variant of a historical charge.

    Transformations are deliberately label-preserving: they alter surface form
    only. The caller retains the original reference-row identity as ground truth.
    """

    synthetic_code = _clean_space(code)
    synthetic_desc = _clean_space(description)
    applied: list[str] = []

    plan = variant_number % 4

    if plan in {0, 3}:
        synthetic_desc, changed = _abbreviate(synthetic_desc, rng)
        if changed:
            applied.append("abbreviation")

    if plan in {1, 3}:
        synthetic_desc, changed = _punctuation_noise(synthetic_desc, rng)
        if changed:
            applied.append("punctuation")

    if plan in {2, 3}:
        synthetic_desc, changed = _drop_low_information_token(synthetic_desc, rng)
        if changed:
            applied.append("token_drop")

    if plan in {1, 2}:
        synthetic_desc, changed = _single_typo(synthetic_desc, rng)
        if changed:
            applied.append("typo")

    if plan in {0, 2}:
        synthetic_code, changed = _code_format_noise(synthetic_code, rng)
        if changed:
            applied.append("code_format")

    if synthetic_code == _clean_space(code) and synthetic_desc == _clean_space(description):
        fallbacks = [
            ("abbreviation", _abbreviate),
            ("punctuation", _punctuation_noise),
            ("token_drop", _drop_low_information_token),
            ("typo", _single_typo),
        ]
        rng.shuffle(fallbacks)
        for name, transform in fallbacks:
            synthetic_desc, changed = transform(synthetic_desc, rng)
            if changed:
                applied.append(name)
                break

    return synthetic_code, synthetic_desc, tuple(applied)


def generate_synthetic_training_data(
    reference: pd.DataFrame,
    *,
    code_col: str = "chrg_code",
    desc_col: str = "chrg_desc",
    year_col: str = "source_year",
    exclude_years: Iterable[int | str] = (),
    config: SyntheticAugmentationConfig | None = None,
) -> pd.DataFrame:
    """Generate label-preserving noisy examples from historical reference rows.

    The original DataFrame index is retained as reference_index and is the
    ground-truth matched row. exclude_years supports leakage-free temporal
    validation, such as training on 2021-2024 while holding out 2025.
    """

    config = config or SyntheticAugmentationConfig()

    required = {code_col, desc_col}
    missing = required - set(reference.columns)
    if missing:
        raise ValueError(f"reference is missing required columns: {sorted(missing)}")

    excluded = {str(year) for year in exclude_years}
    if excluded and year_col not in reference.columns:
        raise ValueError(
            f"exclude_years was provided but year column {year_col!r} is absent"
        )

    rows: list[dict[str, object]] = []

    for row_number, (reference_index, row) in enumerate(reference.iterrows()):
        if excluded and str(row[year_col]) in excluded:
            continue

        reference_code = _clean_space(row[code_col])
        reference_desc = _clean_space(row[desc_col])

        if not reference_desc:
            continue

        for variant_number in range(config.variants_per_row):
            rng = random.Random(
                config.seed + (row_number + 1) * 1009 + (variant_number + 1) * 9176
            )

            synthetic_code, synthetic_desc, transformations = perturb_charge(
                reference_code,
                reference_desc,
                rng=rng,
                variant_number=variant_number,
            )

            rows.append(
                {
                    "reference_index": reference_index,
                    "reference_code": reference_code,
                    "reference_desc": reference_desc,
                    "synthetic_code": synthetic_code,
                    "synthetic_desc": synthetic_desc,
                    "transformations": "|".join(transformations) or "identity",
                    "is_synthetic": True,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "reference_index",
            "reference_code",
            "reference_desc",
            "synthetic_code",
            "synthetic_desc",
            "transformations",
            "is_synthetic",
        ],
    )


def strip_code_formatting(code: str) -> str:
    """Return a canonical alphanumeric code used to verify format-only noise."""

    return re.sub(r"[^A-Za-z0-9]", "", str(code)).upper()
