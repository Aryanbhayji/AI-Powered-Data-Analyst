"""Tests for report formatting helpers."""

import pytest

from src.utils import format_analysis_download


@pytest.mark.parametrize(
    ("file_format", "mime_type", "extension"),
    [("txt", "text/plain", "txt"), ("md", "text/markdown", "md")],
)
def test_report_formats(file_format: str, mime_type: str, extension: str) -> None:
    """Produce downloadable report contents with matching type and extension."""
    content, actual_mime_type, actual_extension = format_analysis_download(
        "Dataset shape: 1 row.", "A short explanation.", file_format
    )

    assert "Dataset shape" in content
    assert "A short explanation" in content
    assert (actual_mime_type, actual_extension) == (mime_type, extension)


def test_unknown_report_format_is_rejected() -> None:
    """Reject output formats the download UI does not support."""
    with pytest.raises(ValueError, match="file_format"):
        format_analysis_download("summary", "explanation", "pdf")

