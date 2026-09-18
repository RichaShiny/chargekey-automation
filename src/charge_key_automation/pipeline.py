"""Backward-compatible jail pipeline imports.

New code should import from ``charge_key_automation.pipelines``.
"""

from .pipelines.jail import JailPipelineSummary, run_jail_pipeline

PipelineSummary = JailPipelineSummary
run_pipeline = run_jail_pipeline

__all__ = ["JailPipelineSummary", "PipelineSummary", "run_jail_pipeline", "run_pipeline"]
