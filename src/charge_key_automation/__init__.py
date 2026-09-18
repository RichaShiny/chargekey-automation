from .config import CourtPipelineConfig, JailPipelineConfig, PipelineConfig
from .pipelines import (
    CourtPipelineSummary,
    JailPipelineSummary,
    run_court_pipeline,
    run_jail_pipeline,
)

__version__ = "0.2.0"

# Backward compatibility with the first app API, where `run_pipeline` meant jail.
run_pipeline = run_jail_pipeline
PipelineSummary = JailPipelineSummary

__all__ = [
    "CourtPipelineConfig",
    "CourtPipelineSummary",
    "JailPipelineConfig",
    "JailPipelineSummary",
    "PipelineConfig",
    "PipelineSummary",
    "run_court_pipeline",
    "run_jail_pipeline",
    "run_pipeline",
]
