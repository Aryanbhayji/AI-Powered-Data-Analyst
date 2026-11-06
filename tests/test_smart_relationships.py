"""Tests for type inference, matching, and automatic relationship analysis."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.command_parser import parse_command, rule_based_plan
from src.plan_executor import execute_plan
from src.plan_validator import PlanValidationError, validate_plan
from src.relationship_analysis import analyze_relationship
from src.type_inference import infer_column_types, match_column_name

TITANIC = Path(__file__).parent / "fixtures" / "titanic_sample.csv"
RELATIONSHIP_COMMANDS = [
    "relationship between ticket fare and survival",
    "ticket fare vs survival",
    "ticket fare versus survival",
    "effect of ticket fare on survival",
    "impact of ticket fare on survival",
    "does ticket fare affect survival",
    "how does ticket fare relate to survival",
    "compare ticket fare across survival",
    "is ticket fare related to survival",
    "correlation between age and fare",
    "age by survival",
    "survival rate by passenger class",
    "compare fare across embarkation ports",
    "is gender related to survival",
    "impact of family size (SibSp) on survival",
    "relationship of fare and survival",
    "explore relationship between fare and survived",
    "what is the relationship between age and pclass",
    "compare age versus class",
    "does fare affect survived",
]


@pytest.fixture
def titanic() -> pd.DataFrame:
    """Load the deterministic local Titanic fixture."""
    return pd.read_csv(TITANIC)


@pytest.mark.parametrize("command", RELATIONSHIP_COMMANDS)
def test_offline_parser_recognizes_relationship_phrasings(
    command: str, titanic: pd.DataFrame
) -> None:
    """Recognize common open-ended relationship wording without a provider."""
    result = rule_based_plan(command, titanic)
    assert result["action"] == "relationship"
    assert result["var_a"] in titanic.columns
    assert result["var_b"] in titanic.columns
    assert result["var_a"] != result["var_b"]


def test_titanic_semantic_types(titanic: pd.DataFrame) -> None:
    """Distinguish binary, discrete, continuous, and identifier-like columns."""
    inferred = infer_column_types(titanic)
    assert inferred["Survived"] == "binary"
    assert inferred["Pclass"] == "numeric_discrete"
    assert inferred["Fare"] == "numeric_continuous"
    assert inferred["Name"] == "text_id"
    assert inferred["Ticket"] == "text_id"
    assert inferred["PassengerId"] == "text_id"


@pytest.mark.parametrize(
    ("mention", "expected"),
    [
        ("ticket fare", "Fare"),
        ("survival", "Survived"),
        ("survivors", "Survived"),
        ("gender", "Sex"),
        ("class", "Pclass"),
        ("Farr", "Fare"),
    ],
)
def test_smart_column_matching(
    mention: str, expected: str, titanic: pd.DataFrame
) -> None:
    """Resolve synonyms, compound phrases, and misspellings to real columns."""
    assert match_column_name(mention, titanic) == expected


def test_relationship_type_pair_analyses_return_charts_and_stats(
    titanic: pd.DataFrame,
) -> None:
    """Cover continuous/grouped, continuous/continuous, and categorical pairs."""
    for left, right, minimum_charts in [
        ("Fare", "Survived", 4),
        ("Age", "Fare", 2),
        ("Sex", "Survived", 2),
    ]:
        result = analyze_relationship(titanic, left, right)
        assert len(result.charts) >= minimum_charts
        assert not result.statistics.empty
        assert "Rows used" in result.statistics["Statistic"].tolist()
        assert "correlation is not causation" in result.explanation.lower()
        for figure in result.charts.values():
            plt.close(figure)


def test_datetime_numeric_pair_is_supported() -> None:
    """Plot a numeric outcome over valid date values."""
    frame = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=30).astype(str),
        "Value": range(30),
    })
    result = analyze_relationship(frame, "Date", "Value")
    assert len(result.charts) >= 2
    assert not result.statistics.empty
    for figure in result.charts.values():
        plt.close(figure)


def test_validator_repairs_missing_relationship_column(titanic: pd.DataFrame) -> None:
    """Keep relationship requests runnable after an unknown column reference."""
    plan = rule_based_plan("relationship between fare and survival", titanic)
    plan["var_a"] = "nonexistent metric"
    validated = validate_plan(plan, titanic)
    assert validated["var_a"] in titanic.columns
    assert validated["warnings"]


def test_validator_repairs_non_numeric_histogram(titanic: pd.DataFrame) -> None:
    """Replace an incompatible numeric chart request with a usable plot."""
    plan = {
        "action": "plot", "plot_type": "histogram", "x": "Sex",
        "y": None, "group_by": None, "aggregation": None, "filters": [],
        "top_n": None, "title": "", "explanation": "",
    }
    result = execute_plan(validate_plan(plan, titanic), titanic)
    assert isinstance(result.figure, matplotlib.figure.Figure)
    assert result.warnings
    plt.close(result.figure)


def test_validator_reports_fuzzy_column_resolution(titanic: pd.DataFrame) -> None:
    """Resolve a plausible spelling mistake and surface a user-facing notice."""
    plan = rule_based_plan("relationship between fare and survival", titanic)
    plan["var_a"] = "Farr"
    validated = validate_plan(plan, titanic)
    assert validated["var_a"] == "Fare"
    assert any("Using 'Fare'" in warning for warning in validated["warnings"])


def test_validator_rejects_invalid_filter_operator(titanic: pd.DataFrame) -> None:
    """Reject unsupported operators rather than interpreting them."""
    plan = {
        "action": "filter", "plot_type": None, "x": None, "y": None,
        "var_a": None, "var_b": None, "group_by": None, "aggregation": None,
        "filters": [{"column": "Age", "op": "contains", "value": 20}],
        "top_n": None, "title": "", "explanation": "",
    }
    with pytest.raises(PlanValidationError, match="operator"):
        validate_plan(plan, titanic)


def test_acceptance_commands_execute_offline_with_multiple_outputs(
    titanic: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify all listed natural-language requests in a provider-free mode."""
    monkeypatch.setenv("AI_PROVIDER", "offline")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    commands = [
        "Relationship between ticket fare and survival",
        "Does age affect survival?",
        "Survival rate by passenger class",
        "Compare fare across embarkation ports",
        "Is gender related to survival?",
        "Correlation between age and fare",
        "Impact of family size (SibSp) on survival",
        "Analyze this dataset",
    ]
    for command in commands:
        plan = parse_command(command, titanic)
        validated = validate_plan(plan, titanic)
        output = execute_plan(validated, titanic)
        assert isinstance(output.figure, matplotlib.figure.Figure)
        assert output.statistics is not None and not output.statistics.empty
        assert output.explanation
        plt.close(output.figure)


def test_plot_requests_produce_one_main_figure(titanic: pd.DataFrame) -> None:
    """Reproduce the missing main plot for relationship commands."""
    plan = rule_based_plan(
        "Relationship between ticket fare and survival", titanic
    )
    result = execute_plan(validate_plan(plan, titanic), titanic)

    assert result.figure is not None
    assert len(result.charts) <= 1
    plt.close(result.figure)


def test_relationship_plot_summary_contains_no_raw_rows(
    titanic: pd.DataFrame,
) -> None:
    """Keep only column and row-count metadata in the explanation prompt."""
    output = execute_plan(
        rule_based_plan("Relationship between ticket fare and survival", titanic),
        titanic,
    )
    assert output.summary
    assert "Synthetic Passenger 001" not in output.summary
    assert "TKT-000" not in output.summary
    assert "Rows used:" in output.summary
    plt.close(output.figure)


def test_small_relationship_datasets_get_friendly_error() -> None:
    """Refuse inferential output when there are fewer than five paired rows."""
    frame = pd.DataFrame({"value": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    with pytest.raises(ValueError, match="5"):
        analyze_relationship(frame, "value", "target")
