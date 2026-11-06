"""Evaluate Ask Your Data visualization coverage on local sample datasets."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.ask_agent import chart_quality_issues, run_analysis
from src.utils import load_project_environment

TITANIC = ROOT / "tests" / "fixtures" / "titanic_sample.csv"
SALES = ROOT / "tests" / "fixtures" / "messy_sales.csv"
REQUESTS = [
    ("titanic", "plot the relationship between ticket fare and survival", None),
    ("titanic", "what is the range of age?", None),
    ("titanic", "heatmap of age, fare and survived", None),
    ("titanic", "compare fare across embarkation ports", None),
    ("titanic", "how spread out is fare by class", None),
    ("titanic", "show survial vs gendr", None),
    ("titanic", "histogram of age", None),
    ("titanic", "distribution of ticket fare", None),
    ("titanic", "survival rate by passenger class", None),
    ("titanic", "relationship between age and fare", None),
    ("titanic", "does age affect survival?", None),
    ("titanic", "effect of class on fare", None),
    ("titanic", "impact of family size SibSp on survival", None),
    ("titanic", "pie chart of passenger class", None),
    ("titanic", "composition of passengers by sex", None),
    ("titanic", "compare age across embarkation ports", None),
    ("titanic", "boxplot of age by sex", None),
    ("titanic", "scatter plot age versus fare", None),
    ("titanic", "correlation heatmap", None),
    ("titanic", "min and max of fare", None),
    ("titanic", "describe passenger age", None),
    ("titanic", "show missing values", None),
    ("titanic", "analyze this dataset", None),
    ("titanic", "now make it a bar chart", "histogram of age"),
    ("titanic", "same but by gender", "relationship between fare and survived"),
    ("sales", "trend of revenue over date", None),
    ("sales", "plot a histogram of units", None),
    ("sales", "what is the range of revenue", None),
    ("sales", "compare revenue across region", None),
    ("sales", "heatmap of units and revenue", None),
    ("sales", "pie chart of product composition", None),
    ("sales", "top categories of product", None),
    ("sales", "relationship between units and revenue", None),
    ("sales", "show units versus discount", None),
    ("sales", "distribution of discount", None),
    ("sales", "range of revenue by region", None),
    ("sales", "average revenue by region as a bar chart", None),
    ("sales", "compare units across channel", None),
    ("sales", "trend revenue by date", None),
    ("sales", "show missing values", None),
    ("sales", "analyze this dataset", None),
    ("sales", "show revenue distribution", None),
    ("sales", "what is min and max of units", None),
    ("sales", "same but by region", "relationship between units and revenue"),
    ("sales", "now make it a bar chart", "histogram of revenue"),
]


def _key_for_provider(provider: str) -> str:
    """Return only the environment variable name for a selected provider."""
    return "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"


def evaluate(live: bool) -> tuple[list[dict[str, Any]], float]:
    """Run all requests and return per-request records and the visualization error rate."""
    load_project_environment()
    provider = os.getenv("AI_PROVIDER", "openai").lower()
    api_key = os.getenv(_key_for_provider(provider), "")
    if live and not api_key:
        raise RuntimeError(
            "Live evaluation was requested, but no provider API key is configured."
        )
    live_enabled = live and bool(api_key)
    frames = {
        "titanic": pd.read_csv(TITANIC),
        "sales": pd.read_csv(SALES),
    }
    records: list[dict[str, Any]] = []
    failures = 0
    expected_plots = 0
    for index, (dataset, request, previous) in enumerate(REQUESTS, start=1):
        frame = frames[dataset]
        result = None
        error = ""
        try:
            result = run_analysis(
                request,
                frame,
                previous_request=previous,
                provider=provider,
                api_key=api_key if live_enabled else None,
                use_model=live_enabled,
            )
            figures = result.figures
            valid = bool(figures) and not chart_quality_issues(
                figures[0], frame, result.plan.get("resolved_columns", [])
            )
            if not valid:
                error = "No valid titled and labeled figure."
        except Exception as exc:
            figures = []
            valid = False
            error = str(exc)
        expected_plots += 1
        failures += not valid
        records.append({
            "id": index,
            "dataset": dataset,
            "request": request,
            "success": bool(valid),
            "retries": result.retries if result else 0,
            "figures": len(figures),
            "title_exists": bool(
                figures and any(axis.get_title() for axis in figures[0].axes)
            ),
            "axis_labels_exist": bool(
                figures and any(
                    axis.get_xlabel() or axis.get_ylabel()
                    for axis in figures[0].axes
                )
            ),
            "error": error,
            "mode": "live" if live_enabled else "offline",
        })
        if result:
            for figure in result.figures:
                plt.close(figure)
    return records, 100.0 * failures / max(1, expected_plots)


def main() -> None:
    """Run evaluation and write a CSV report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline", action="store_true",
        help="Disable provider calls and use the offline planner/explainer.",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "evaluation" / "results.csv",
        help="Path for the per-request CSV output.",
    )
    arguments = parser.parse_args()
    load_project_environment()
    provider = os.getenv("AI_PROVIDER", "openai").lower()
    has_key = bool(os.getenv(_key_for_provider(provider)))
    live = has_key and not arguments.offline
    if not has_key and not arguments.offline:
        print(
            "No provider API key is configured; running offline evaluation. "
            "For live evaluation, set OPENAI_API_KEY or ANTHROPIC_API_KEY."
        )
    records, error_rate = evaluate(live=live)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Requests: {len(records)}")
    print(f"Visualization Error Rate: {error_rate:.2f}%")
    print(f"Results CSV: {arguments.output}")
    if live:
        print("Evaluation mode: live provider")
    else:
        print("Evaluation mode: offline (not a live-LLM accuracy measurement)")


if __name__ == "__main__":
    main()
