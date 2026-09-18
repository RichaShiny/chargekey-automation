from __future__ import annotations

import argparse

from .config import CourtPipelineConfig, JailPipelineConfig
from .pipelines import run_court_pipeline, run_jail_pipeline
from .sites import COURT_SITES, JAIL_SITES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ISLG no-LLM charge-key automation for jail and court workflows"
    )
    subparsers = parser.add_subparsers(dest="workflow", required=True)

    jail = subparsers.add_parser("jail", help="Update a jail charge key from a diagnostic report")
    _common_arguments(jail, JAIL_SITES)
    jail.add_argument("--rerank-threshold", type=float, default=0.85)
    jail.add_argument("--margin-threshold", type=float, default=0.03)

    court = subparsers.add_parser(
        "court", help="Match court charges against the finalized jail charge key"
    )
    _common_arguments(court, COURT_SITES)
    court.add_argument("--agreement-threshold", type=float, default=0.80)
    court.add_argument("--probability-threshold", type=float, default=0.85)
    return parser


def _common_arguments(parser: argparse.ArgumentParser, sites: tuple[str, ...]) -> None:
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--site", required=True, choices=sites)
    parser.add_argument("--year", default="2026")
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument("--fuzzy-threshold", type=int, default=90)


def main() -> None:
    args = build_parser().parse_args()
    if args.workflow == "jail":
        summary = run_jail_pipeline(
            args.data_dir,
            args.output_dir,
            JailPipelineConfig(
                site_name=args.site,
                target_year=args.year,
                similarity_threshold=args.similarity_threshold,
                fuzzy_threshold=args.fuzzy_threshold,
                rerank_threshold=args.rerank_threshold,
                margin_threshold=args.margin_threshold,
            ),
        )
        print(f"Saved: {summary.output_path}")
        print(
            f"Auto-classified: {summary.auto_classified}/{summary.total_new_charges} "
            f"({summary.automation_rate}%)"
        )
        print(f"Needs review: {summary.needs_review}")
    else:
        summary = run_court_pipeline(
            args.data_dir,
            args.output_dir,
            CourtPipelineConfig(
                site_name=args.site,
                target_year=args.year,
                similarity_threshold=args.similarity_threshold,
                fuzzy_threshold=args.fuzzy_threshold,
                agreement_threshold=args.agreement_threshold,
                probability_threshold=args.probability_threshold,
            ),
        )
        print(f"Saved: {summary.output_path}")
        print(f"Confident: {summary.confident}")
        print(f"Ensemble resolved: {summary.ensemble_resolved}")
        print(f"Needs review: {summary.needs_review}")
        print(f"Automation rate: {summary.automation_rate}%")


if __name__ == "__main__":
    main()
