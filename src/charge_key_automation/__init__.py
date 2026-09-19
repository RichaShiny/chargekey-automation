from .config import CourtPipelineConfig, JailPipelineConfig, PipelineConfig
from .pipelines import (
    CourtPipelineSummary,
    JailPipelineSummary,
    run_court_pipeline,
    run_jail_pipeline,
)
from .pairwise_reranker import (
    PairwiseReranker,
    PairwiseRerankerEvaluation,
    build_pairwise_training_data,
    evaluate_pairwise_reranker,
    fit_pairwise_reranker,
    rerank_query_candidates,
)
from .retrieval_eval import (
    RetrievalEvaluation,
    RetrievalEvaluationConfig,
    evaluate_candidate_retrieval,
    mine_hard_negatives,
    rank_query_candidates,
)
from .synthetic import SyntheticAugmentationConfig, generate_synthetic_training_data
from .temporal_validation import (
    TemporalValidationConfig,
    TemporalValidationResult,
    evaluate_temporal_holdout,
    fit_temporal_reranker,
)

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
    "PairwiseReranker",
    "PairwiseRerankerEvaluation",
    "RetrievalEvaluation",
    "RetrievalEvaluationConfig",
    "SyntheticAugmentationConfig",
    "TemporalValidationConfig",
    "TemporalValidationResult",
    "build_pairwise_training_data",
    "evaluate_pairwise_reranker",
    "fit_pairwise_reranker",
    "fit_temporal_reranker",
    "evaluate_candidate_retrieval",
    "evaluate_temporal_holdout",
    "mine_hard_negatives",
    "rank_query_candidates",
    "rerank_query_candidates",
    "generate_synthetic_training_data",
    "run_court_pipeline",
    "run_jail_pipeline",
    "run_pipeline",
]
