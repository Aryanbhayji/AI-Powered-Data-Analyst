"""Infer semantic column roles for automatic dataset exploration."""

from __future__ import annotations

import re

import pandas as pd

COLUMN_TYPES = {
    "numeric_continuous",
    "numeric_discrete",
    "binary",
    "categorical",
    "datetime",
    "text_id",
}

_DATE_PATTERN = re.compile(
    r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
)


def _parse_datetime(series: pd.Series) -> pd.Series:
    """Parse likely date strings while avoiding noisy parsing warnings."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    values = series.dropna().astype(str)
    if values.empty or values.str.match(_DATE_PATTERN).mean() < 0.8:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    return pd.to_datetime(series, errors="coerce", format="mixed")


def infer_column_type(series: pd.Series, name: str = "") -> str:
    """Classify a column as numeric, binary, categorical, datetime, or ID-like."""
    values = series.dropna()
    if values.empty:
        return "categorical"

    unique_count = int(values.nunique())
    unique_ratio = unique_count / len(values)
    normalized_name = re.sub(r"[^a-z0-9]", "", name.casefold())
    string_dtype = (
        pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )
    if normalized_name.endswith(("id", "ticket", "name")) and unique_ratio > 0.5:
        return "text_id"
    if string_dtype and _parse_datetime(series).notna().mean() >= 0.8:
        return "datetime"

    numeric = pd.to_numeric(values, errors="coerce")
    if float(numeric.notna().mean()) >= 0.8:
        distinct = int(numeric.nunique())
        if distinct == 2:
            return "binary"
        if unique_ratio > 0.9 and string_dtype:
            return "text_id"
        if unique_ratio > 0.9 and normalized_name.endswith(("id", "number")):
            return "text_id"
        if distinct <= 10:
            return "numeric_discrete"
        return "numeric_continuous"

    if unique_count == 2:
        return "binary"
    if unique_ratio > 0.9 and string_dtype:
        return "text_id"
    return "categorical"


def infer_column_types(dataframe: pd.DataFrame) -> dict[str, str]:
    """Return a semantic type for every dataframe column."""
    return {
        str(column): infer_column_type(dataframe[column], str(column))
        for column in dataframe.columns
    }
