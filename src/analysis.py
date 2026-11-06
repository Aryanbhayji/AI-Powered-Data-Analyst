"""Data profiling operations that do not depend on the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass
class AnalysisResult:
    """Tables and metrics calculated for a dataframe."""

    overview: dict[str, Any]
    missing_values: pd.DataFrame
    numeric_summary: pd.DataFrame
    categorical_summary: dict[str, dict[str, Any]]
    skewness: pd.Series
    outliers: pd.DataFrame
    correlations: pd.DataFrame


def calculate_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and percentages, ordered by missing percentage."""
    total_rows = len(dataframe)
    counts = dataframe.isna().sum()
    percentages = counts.div(total_rows).mul(100) if total_rows else counts.astype(float)
    result = pd.DataFrame({
        "Column": dataframe.columns.astype(str),
        "Missing Count": counts.to_numpy(dtype=int),
        "Missing %": percentages.to_numpy(dtype=float),
    })
    return result.sort_values(
        ["Missing %", "Column"], ascending=[False, True], ignore_index=True
    )


def calculate_outlier_counts(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Count numeric values outside each column's 1.5-IQR range."""
    records: list[dict[str, int | float | str]] = []
    for column in dataframe.columns:
        series = dataframe[column]
        if not is_numeric_dtype(series):
            continue
        values = series.dropna()
        if values.empty:
            outlier_count = 0
        else:
            first_quartile = values.quantile(0.25)
            third_quartile = values.quantile(0.75)
            spread = third_quartile - first_quartile
            lower_bound = first_quartile - 1.5 * spread
            upper_bound = third_quartile + 1.5 * spread
            outlier_count = int(((values < lower_bound) | (values > upper_bound)).sum())
        denominator = int(series.notna().sum())
        records.append({
            "Column": str(column),
            "Outlier Count": outlier_count,
            "Outlier %": (outlier_count / denominator * 100) if denominator else 0.0,
        })
    return pd.DataFrame(records, columns=["Column", "Outlier Count", "Outlier %"])


def analyze_dataframe(dataframe: pd.DataFrame) -> AnalysisResult:
    """Calculate overview, missingness, statistics, correlations, and outliers."""
    numeric_columns = [
        column for column in dataframe.columns if is_numeric_dtype(dataframe[column])
    ]
    categorical_columns = [column for column in dataframe.columns
                           if column not in numeric_columns]
    numeric = dataframe[numeric_columns]
    numeric_summary = numeric.describe().transpose() if numeric_columns else pd.DataFrame()
    skewness = numeric.skew(numeric_only=True) if numeric_columns else pd.Series(dtype=float)

    categorical_summary: dict[str, dict[str, Any]] = {}
    for column in categorical_columns:
        series = dataframe[column]
        values = series.dropna().astype(str).value_counts()
        categorical_summary[str(column)] = {
            "unique_count": int(series.nunique(dropna=True)),
            "top_values": values.head(10),
        }

    correlations = numeric.corr() if len(numeric_columns) >= 2 else pd.DataFrame()
    overview = {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "memory_bytes": int(dataframe.memory_usage(index=True, deep=True).sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "dtypes": {str(column): str(dtype)
                   for column, dtype in dataframe.dtypes.items()},
    }

    return AnalysisResult(
        overview=overview,
        missing_values=calculate_missing_values(dataframe),
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        skewness=skewness.replace([np.inf, -np.inf], np.nan),
        outliers=calculate_outlier_counts(dataframe),
        correlations=correlations,
    )

