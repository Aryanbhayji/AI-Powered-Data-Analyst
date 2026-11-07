"""Safe CSV loading and validation helpers."""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import BinaryIO

import pandas as pd

LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


class CSVLoadError(ValueError):
    """Raised when an uploaded file cannot be safely loaded as a CSV."""


def validate_upload(filename: str, size_bytes: int) -> None:
    """Reject uploads with an unsupported extension or excessive size."""
    if Path(filename).suffix.lower() != ".csv":
        raise CSVLoadError("Please upload a file with the .csv extension.")
    if size_bytes > MAX_UPLOAD_BYTES:
        raise CSVLoadError("The CSV exceeds the 50 MB upload limit.")
    if size_bytes < 0:
        raise CSVLoadError("The uploaded file size is invalid.")


def _read_bytes(
    source: str | Path | BinaryIO | bytes, max_size_bytes: int
) -> bytes:
    """Read a supported input source into bytes."""
    try:
        if isinstance(source, bytes):
            return source
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.stat().st_size > max_size_bytes:
                raise CSVLoadError("The CSV exceeds the configured upload limit.")
            with path.open("rb") as stream:
                return stream.read(max_size_bytes + 1)
        content = source.read(max_size_bytes + 1)
        if isinstance(content, str):
            return content.encode("utf-8")
        return content
    except CSVLoadError:
        raise
    except OSError as exc:
        raise CSVLoadError(f"Unable to read the uploaded file: {exc}") from exc


def _detect_delimiter(sample: str) -> str:
    """Detect comma, semicolon, or tab separators from a text sample."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        return dialect.delimiter
    except csv.Error:
        lines = [line for line in sample.splitlines()[:20] if line.strip()]
        counts = {delimiter: sum(line.count(delimiter) for line in lines)
                  for delimiter in (",", ";", "\t")}
        return max(counts, key=counts.get) if max(counts.values(), default=0) else ","


def _unique_column_names(columns: pd.Index) -> list[str]:
    """Ensure column names are unique without discarding their original names."""
    used: set[str] = set()
    result: list[str] = []
    for position, value in enumerate(columns):
        name = str(value)
        candidate = name
        suffix = 1
        while candidate in used:
            candidate = f"{name}.{suffix}"
            suffix += 1
        if not candidate:
            candidate = f"column_{position + 1}"
            while candidate in used:
                suffix += 1
                candidate = f"column_{position + 1}.{suffix}"
        used.add(candidate)
        result.append(candidate)
    return result


def load_csv(
    source: str | Path | BinaryIO | bytes,
    filename: str | None = None,
    max_size_bytes: int = MAX_UPLOAD_BYTES,
) -> pd.DataFrame:
    """Load a CSV with encoding/delimiter detection and bounded input size."""
    content = _read_bytes(source, max_size_bytes)
    if filename is not None:
        validate_upload(filename, len(content))
    if len(content) > max_size_bytes:
        raise CSVLoadError("The CSV exceeds the configured upload limit.")
    if not content.strip():
        raise CSVLoadError("The uploaded CSV is empty.")

    text: str | None = None
    for encoding in ENCODINGS:
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise CSVLoadError("The CSV encoding is not supported.")

    delimiter = _detect_delimiter(text[:65536])
    skipped_rows: list[int] = []

    def skip_malformed_row(fields: list[str]) -> None:
        """Log malformed rows encountered by pandas' Python parser."""
        skipped_rows.append(1)
        LOGGER.warning("Skipping malformed CSV row with %d fields.", len(fields))

    try:
        frame = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            engine="python",
            on_bad_lines=skip_malformed_row,
        )
    except pd.errors.EmptyDataError as exc:
        raise CSVLoadError("The uploaded CSV does not contain a header row.") from exc
    except (pd.errors.ParserError, UnicodeError, csv.Error) as exc:
        raise CSVLoadError(f"The CSV could not be parsed: {exc}") from exc

    if len(frame.columns) == 0:
        raise CSVLoadError("The uploaded CSV does not contain any columns.")
    frame.columns = _unique_column_names(frame.columns)
    frame.attrs["malformed_rows_skipped"] = len(skipped_rows)
    return frame
