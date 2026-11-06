"""Tests for safe CSV loading."""

from pathlib import Path

import pandas as pd
import pytest

from src.loader import CSVLoadError, load_csv

FIXTURES = Path(__file__).parent / "fixtures"


def test_loads_valid_fixture_csv() -> None:
    """Load a standard UTF-8 CSV and preserve its shape."""
    dataframe = load_csv(FIXTURES / "loader_good.csv")

    assert dataframe.shape == (3, 3)
    assert dataframe["name"].tolist() == ["Ada", "Grace", "Linus"]


def test_loads_latin_encoding_and_semicolon_delimiter() -> None:
    """Decode a Western-encoded file and detect semicolon separators."""
    contents = b"name;city\nAndr\xe9;Montr\xe9al\n"
    dataframe = load_csv(contents)

    assert dataframe.iloc[0].to_dict() == {"name": "André", "city": "Montréal"}


def test_empty_file_raises_friendly_error() -> None:
    """Reject zero-byte input instead of exposing a pandas traceback."""
    with pytest.raises(CSVLoadError, match="empty"):
        load_csv(b"")


def test_header_only_and_single_column_files_are_supported() -> None:
    """Keep legitimate header-only and one-column files loadable."""
    header_only = load_csv(b"only_column\n")
    single_column = load_csv(b"only_column\nfirst\nsecond\n")

    assert header_only.empty
    assert list(single_column.columns) == ["only_column"]
    assert single_column["only_column"].tolist() == ["first", "second"]


def test_upload_extension_and_size_are_validated() -> None:
    """Reject non-CSV names and inputs beyond the configured byte limit."""
    with pytest.raises(CSVLoadError, match=".csv"):
        load_csv(b"a\n1\n", filename="data.txt")
    with pytest.raises(CSVLoadError, match="configured upload limit"):
        load_csv(b"a\n1\n", filename="data.csv", max_size_bytes=1)


def test_duplicate_header_names_are_made_unique() -> None:
    """Avoid ambiguous duplicate column labels."""
    dataframe = load_csv(b"value,value\n1,2\n")

    assert dataframe.columns.is_unique
    assert isinstance(dataframe, pd.DataFrame)


def test_malformed_rows_are_skipped_and_reported() -> None:
    """Skip rows with too many fields and expose a count for the UI warning."""
    dataframe = load_csv(b"name,score\nAda,98\nBroken,row,extra\nGrace,95\n")

    assert dataframe.shape == (2, 2)
    assert dataframe.attrs["malformed_rows_skipped"] == 1
