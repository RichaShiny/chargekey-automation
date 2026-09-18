from __future__ import annotations

from dataclasses import dataclass, field

from .sites import COURT_SITES, JAIL_SITES


@dataclass(frozen=True)
class JailPipelineConfig:
    site_name: str
    target_year: str = "2026"
    similarity_threshold: float = 0.85
    fuzzy_threshold: int = 90
    rerank_threshold: float = 0.85
    margin_threshold: float = 0.03
    top_k: int = 5
    similarity_weight: float = 0.6
    attribute_weight: float = 0.4
    min_class_count: int = 5
    reference_years: tuple[str, ...] = field(default_factory=tuple)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    def resolved_reference_years(self) -> tuple[str, ...]:
        if self.reference_years:
            return tuple(str(year) for year in self.reference_years)
        year = int(self.target_year)
        return tuple(str(value) for value in range(year - 5, year))

    def validate(self) -> None:
        if self.site_name not in JAIL_SITES:
            raise ValueError(f"Unsupported jail jurisdiction: {self.site_name}")
        _validate_thresholds(
            self.similarity_threshold,
            self.fuzzy_threshold,
            self.rerank_threshold,
            self.margin_threshold,
        )


@dataclass(frozen=True)
class CourtPipelineConfig:
    site_name: str
    target_year: str = "2026"
    similarity_threshold: float = 0.85
    fuzzy_threshold: int = 90
    agreement_threshold: float = 0.80
    probability_threshold: float = 0.85
    min_class_count: int = 5
    reference_years: tuple[str, ...] = field(default_factory=tuple)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    def resolved_reference_years(self) -> tuple[str, ...]:
        if self.reference_years:
            return tuple(str(year) for year in self.reference_years)
        # Original 2026 court workflow trained on 2021-2024 jail keys.
        year = int(self.target_year)
        return tuple(str(value) for value in range(year - 5, year - 1))

    def validate(self) -> None:
        if self.site_name not in COURT_SITES:
            raise ValueError(f"Unsupported court jurisdiction: {self.site_name}")
        _validate_thresholds(self.similarity_threshold, self.fuzzy_threshold)
        if not 0 <= self.agreement_threshold <= 1:
            raise ValueError("agreement_threshold must be between 0 and 1")
        if not 0 <= self.probability_threshold <= 1:
            raise ValueError("probability_threshold must be between 0 and 1")


# Backward-compatible alias for the original desktop/jail API.
PipelineConfig = JailPipelineConfig


def _validate_thresholds(
    similarity_threshold: float,
    fuzzy_threshold: int,
    rerank_threshold: float | None = None,
    margin_threshold: float | None = None,
) -> None:
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")
    if not 0 <= fuzzy_threshold <= 100:
        raise ValueError("fuzzy_threshold must be between 0 and 100")
    if rerank_threshold is not None and not 0 <= rerank_threshold <= 1:
        raise ValueError("rerank_threshold must be between 0 and 1")
    if margin_threshold is not None and margin_threshold < 0:
        raise ValueError("margin_threshold cannot be negative")
