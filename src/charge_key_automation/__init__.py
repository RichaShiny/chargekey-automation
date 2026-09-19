from .config import CourtPipelineConfig, JailPipelineConfig, PipelineConfig
from .pipelines import (
    CourtPipelineSummary,
    JailPipelineSummary,
    run_court_pipeline,
    run_jail_pipeline,
)
from .retrieval_eval import (
    RetrievalEvaluation,
    RetrievalEvaluationConfig,
    evaluate_candidate_retrieval,
    mine_hard_negatives,
    rank_query_candidates,
)
from .synthetic import SyntheticAugmentationConfig, generate_synthetic_training_data

__version__ = "0.2.0"

# Backward compatibility with the first app API, where run_pipeline meant jail.
run_pipeline = run_jail_pipeline
PipelineSummary = JailPipelineSummary

__all__ = [
    "CourtPipelineConfig",
    "CourtPipelineSummary",
    "JailPipelineConfig",
    "JailPipelineSummary",
    "PipelineConfig",
    "PipelineSummary",
    "RetrievalEvaluation",
    "RetrievalEvaluationConfig",
    "SyntheticAugmentationConfig",
    "evaluate_candidate_retrieval",
    "mine_hard_negatives",
    "rank_query_candidates",
    "generate_synthetic_training_data",
    "run_court_pipeline",
    "run_jail_pipeline",
    "run_pipeline",
]
