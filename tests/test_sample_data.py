"""Smoke tests for the shipped demo datasets."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src.ai_explainer import build_analysis_summary
from src.analysis import analyze_dataframe
from src.command_parser import parse_command
from src.loader import load_csv
from src.plan_executor import execute_plan
from src.plan_validator import validate_plan

SAMPLE_DATA = Path(__file__).parents[1] / "sample_data"


def test_both_demo_files_load_and_analyze() -> None:
    """Make sure clean and messy demos are consumable by the analysis pipeline."""
    for filename in ("clean_sales.csv", "messy_survey.csv"):
        dataframe = load_csv(SAMPLE_DATA / filename)
        analysis = analyze_dataframe(dataframe)
        summary = build_analysis_summary(analysis)

        assert not dataframe.empty
        assert analysis.overview["columns"] == dataframe.shape[1]
        assert f"{dataframe.shape[0]} rows" in summary


def test_offline_commands_work_on_both_demo_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run over 10 real sample-data requests through parse/validate/execute."""
    def no_provider(command: str, schema: dict[str, object]) -> str:
        raise RuntimeError("Tests use the offline command parser.")

    monkeypatch.setattr("src.command_parser._invoke_model", no_provider)
    sample_commands = {
        "clean_sales.csv": [
            "plot a histogram of units",
            "scatter plot units vs revenue",
            "show correlation between units and revenue",
            "average revenue by region as a bar chart",
            "sum units by product",
            "top 3 region by count",
            "show rows where units > 10",
            "describe revenue",
            "pie chart of product",
            "boxplot of revenue by region",
            "count rows",
        ],
        "messy_survey.csv": [
            "plot a histogram of age",
            "scatter plot age vs satisfaction",
            "show correlation between age and satisfaction",
            "average satisfaction by department as a bar chart",
            "top 2 department by count",
            "show rows where age > 30",
            "describe age",
            "pie chart of department",
            "boxplot of satisfaction by department",
            "count rows",
        ],
    }

    for filename, commands in sample_commands.items():
        dataframe = load_csv(SAMPLE_DATA / filename)
        for command in commands:
            plan = validate_plan(parse_command(command, dataframe), dataframe)
            result = execute_plan(plan, dataframe)
            assert result.figure is not None or result.table is not None or result.text
            if result.figure is not None:
                plt.close(result.figure)
            for figure in result.charts.values():
                plt.close(figure)
