"""Tests for chart builders."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.visualizations import (
    boxplot,
    category_bar,
    correlation_heatmap,
    histogram,
    missingness_bar_chart,
)


def test_chart_builders_return_figures_for_regular_and_degenerate_data() -> None:
    """Build requested charts even when there are no numeric or missing values."""
    frame = pd.DataFrame({
        "number": [1, 2, 3, 100],
        "other": [2, 4, 6, 8],
        "category": ["a", "b", "a", None],
    })
    figures = [
        histogram(frame, "number"),
        boxplot(frame, "number"),
        category_bar(frame, "category"),
        correlation_heatmap(frame),
        missingness_bar_chart(pd.DataFrame({
            "Column": ["category"], "Missing Count": [1], "Missing %": [25.0]
        })),
        correlation_heatmap(pd.DataFrame({"label": ["x", "y"]})),
    ]

    assert all(isinstance(figure, Figure) for figure in figures)
    for figure in figures:
        plt.close(figure)


def test_high_cardinality_categories_are_truncated() -> None:
    """Limit category charts to the requested top-N values."""
    frame = pd.DataFrame({"category": [f"item-{index}" for index in range(100)]})
    figure = category_bar(frame, "category", top_n=5)

    assert len(figure.axes[0].patches) == 5
    plt.close(figure)
