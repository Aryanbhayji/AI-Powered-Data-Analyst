"""Smoke tests for the shipped demo datasets."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ai_explainer import build_analysis_summary
from src.analysis import analyze_dataframe
from src.auto_visualizer import build_visualization_report
from src.loader import load_csv

SAMPLE_DATA = Path(__file__).parents[1] / "sample_data"


def test_both_demo_files_load_analyze_and_visualize() -> None:
    """Make sure clean and messy demos support the primary analysis workflow."""
    for filename in ("clean_sales.csv", "messy_survey.csv"):
        dataframe = load_csv(SAMPLE_DATA / filename)
        analysis = analyze_dataframe(dataframe)
        summary = build_analysis_summary(analysis)
        report = build_visualization_report(dataframe)

        assert not dataframe.empty
        assert analysis.overview["columns"] == dataframe.shape[1]
        assert f"{dataframe.shape[0]} rows" in summary
        assert report.insights
        assert report.charts
        for chart in report.charts:
            assert chart.figure.axes
            plt.close(chart.figure)
