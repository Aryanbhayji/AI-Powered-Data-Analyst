"""Tests for dataframe profiling and statistics."""

import numpy as np
import pandas as pd

from src.analysis import (
    analyze_dataframe,
    calculate_missing_values,
    calculate_outlier_counts,
)


def test_missing_value_counts_and_percentages_are_sorted() -> None:
    """Calculate missing counts and percentages in descending order."""
    dataframe = pd.DataFrame({"few": [1, None, 3, 4], "many": [None] * 4})

    result = calculate_missing_values(dataframe)

    assert result["Column"].tolist() == ["many", "few"]
    assert result.loc[0, "Missing Count"] == 4
    assert result.loc[1, "Missing %"] == 25.0


def test_mixed_type_statistics_and_categorical_uniques() -> None:
    """Analyze numeric and categorical columns without coercing either."""
    dataframe = pd.DataFrame({
        "amount": [1.0, 2.0, 3.0, 4.0],
        "label": ["a", "a", None, "b"],
        "mixed": ["1", "two", "3", "four"],
    })

    result = analyze_dataframe(dataframe)

    assert result.numeric_summary.loc["amount", "mean"] == 2.5
    assert result.categorical_summary["label"]["unique_count"] == 2
    assert result.categorical_summary["mixed"]["unique_count"] == 4
    assert result.overview["rows"] == 4


def test_iqr_outliers_are_counted_per_numeric_column() -> None:
    """Count a distant value as an IQR outlier and ignore nulls."""
    dataframe = pd.DataFrame({"measure": [1, 2, 2, 3, 100, np.nan]})

    result = calculate_outlier_counts(dataframe)

    assert result.loc[0, "Outlier Count"] == 1
    assert result.loc[0, "Outlier %"] == 20.0


def test_empty_numeric_set_and_all_null_column_are_safe() -> None:
    """Handle entirely categorical frames and null numeric columns."""
    categorical = analyze_dataframe(pd.DataFrame({"when": ["2025-01-01", "2025-01-02"]}))
    all_null = analyze_dataframe(pd.DataFrame({"value": [np.nan, np.nan]}))

    assert categorical.numeric_summary.empty
    assert categorical.correlations.empty
    assert all_null.missing_values.loc[0, "Missing %"] == 100.0
    assert all_null.outliers.loc[0, "Outlier Count"] == 0


def test_single_row_and_no_columns_do_not_raise() -> None:
    """Support degenerate dataframes used in defensive callers."""
    one_row = analyze_dataframe(pd.DataFrame({"value": [7]}))
    no_columns = analyze_dataframe(pd.DataFrame(index=[0]))

    assert one_row.overview["rows"] == 1
    assert no_columns.overview["columns"] == 0

