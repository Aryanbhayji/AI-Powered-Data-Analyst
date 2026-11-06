"""Small formatting helpers shared by the application."""

from __future__ import annotations

from dotenv import find_dotenv, load_dotenv


def load_project_environment() -> None:
    """Load a project-local .env safely from scripts, tests, and Streamlit."""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)


def format_analysis_download(
    summary: str, explanation: str, file_format: str = "txt"
) -> tuple[str, str, str]:
    """Return download content, MIME type, and extension for TXT or Markdown."""
    selected_format = file_format.lower().lstrip(".")
    if selected_format not in {"txt", "md"}:
        raise ValueError("file_format must be 'txt' or 'md'.")
    if selected_format == "md":
        content = f"# AI Data Analyst report\n\n## Analysis summary\n\n{summary}\n\n"
        content += f"## Plain-English explanation\n\n{explanation}\n"
        return content, "text/markdown", "md"
    content = f"AI Data Analyst report\n\nAnalysis summary\n----------------\n{summary}\n\n"
    content += f"Plain-English explanation\n-------------------------\n{explanation}\n"
    return content, "text/plain", "txt"
