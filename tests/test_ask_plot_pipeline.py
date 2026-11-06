"""End-to-end tests for visible Ask Your Data plots."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure
from streamlit.testing.v1 import AppTest

import app
from src.command_parser import parse_command
from src.plan_executor import execute_plan
from src.plan_validator import validate_plan
from src.visualizations import (
    command_boxplot,
    command_countplot,
    command_heatmap,
    command_histogram,
    command_lineplot,
    command_pairplot,
    command_pie,
    command_rate_bar,
    command_scatter,
    command_violinplot,
)

FIXTURE = Path(__file__).parent / "fixtures" / "titanic_sample.csv"
APP_PATH = Path(__file__).parents[1] / "app.py"
NEW_ACCEPTANCE = [
    "plot the relationship between fare and survival",
    "heatmap of age, fare and survived",
    "correlation heatmap",
    "relationship between age and fare",
    "heatmap of all features",
    "show survival vs passenger class",
    "pairplot of age, fare, pclass",
    "plot ticket fare against survival",
    "heat map survial and fair",
    "scatter plot of age and fare",
]
ACCEPTANCE = [
    ("Relationship between ticket fare and survival", "box", "Fare", "Survived",
     "Fare by Survived"),
    ("Plot age vs fare", "scatter", "Age", "Fare", "Fare vs Age"),
    ("Survival rate by passenger class", "bar", "Pclass", "Survived",
     "Mean Survived by Pclass"),
    ("Histogram of age", "histogram", "Age", None, "Distribution of Age"),
    ("Is gender related to survival?", "count", "Sex", "Survived",
     "Counts of Sex by Survived"),
    ("Show correlation heatmap", "heatmap", None, None,
     "Numeric correlation heatmap"),
    ("Compare fare across embarkation ports", "box", "Fare", "Embarked",
     "Fare by Embarked"),
    ("Pie chart of passenger class", "pie", "Pclass", None,
     "Distribution of Pclass"),
    ("Boxplot of age by Sex", "box", "Age", "Sex", "Age by Sex"),
    ("Does age affect survival?", "box", "Age", "Survived",
     "Age by Survived"),
]


@pytest.fixture
def titanic() -> pd.DataFrame:
    """Load the local Titanic-like fixture."""
    return pd.read_csv(FIXTURE)


def _close_result(result: object) -> None:
    """Close the one returned matplotlib figure."""
    figure = getattr(result, "figure", None)
    if isinstance(figure, Figure):
        plt.close(figure)


@pytest.mark.parametrize("command,chart,x,y,expected_title", ACCEPTANCE)
def test_acceptance_pipeline_always_returns_one_figure(
    command: str, chart: str, x: str | None, y: str | None, expected_title: str,
    titanic: pd.DataFrame, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run every acceptance command from parsing through drawing."""
    monkeypatch.setenv("AI_PROVIDER", "offline")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    plan = parse_command(
        command, titanic,
        model_caller=lambda *_: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    result = execute_plan(validate_plan(plan, titanic), titanic)

    assert isinstance(result.figure, Figure)
    assert len(result.figure.axes) > 0
    axis = result.figure.axes[0]
    assert axis.get_title() == expected_title
    if chart not in {"heatmap", "pie"}:
        assert axis.get_xlabel()
        assert axis.get_ylabel()
    assert result.statistics is not None
    assert int(result.statistics.loc[
        result.statistics["Statistic"] == "Rows used", "Value"
    ].iloc[0]) > 0
    if command.startswith("Relationship") or command.startswith("Compare"):
        assert x in axis.get_title() or x in axis.get_xlabel() or x in axis.get_ylabel()
    _close_result(result)


def test_misspelled_column_names_are_repaired(titanic: pd.DataFrame) -> None:
    """Resolve common typos in both sides of a relationship."""
    from src.command_parser import rule_based_plan

    plan = rule_based_plan("Relationship between survial and fair", titanic)
    assert plan["action"] == "relationship"
    assert {plan["var_a"], plan["var_b"]} == {"Survived", "Fare"}
    result = execute_plan(validate_plan(plan, titanic), titanic)
    assert isinstance(result.figure, Figure)
    assert "Fare" in result.figure.axes[0].get_title()
    plt.close(result.figure)


def test_chart_builders_return_labeled_figures(titanic: pd.DataFrame) -> None:
    """Exercise all individual chart functions used by the command dispatcher."""
    builders = [
        command_histogram(titanic, "Age"),
        command_scatter(titanic, "Age", "Fare"),
        command_boxplot(titanic, "Fare", "Survived"),
        command_countplot(titanic, "Sex", "Survived"),
        command_rate_bar(titanic, "Survived", "Pclass"),
        command_lineplot(titanic, "Age", "Fare"),
        command_pie(titanic, "Pclass"),
        command_heatmap(titanic),
        command_violinplot(titanic, "Age", "Sex"),
        command_pairplot(titanic[["Age", "Fare", "SibSp", "Parch"]]),
    ]
    for figure in builders:
        assert isinstance(figure, Figure)
        assert figure.axes and figure.axes[0].get_title()
        assert any(axis.get_xlabel() or axis.get_ylabel() for axis in figure.axes)
        plt.close(figure)


def test_invalid_model_json_falls_back_to_rule_parser(
    titanic: pd.DataFrame,
) -> None:
    """Retry invalid AI output then return a usable offline chart plan."""
    attempts: list[str] = []

    def invalid(command: str, _schema: dict[str, object]) -> str:
        attempts.append(command)
        return "not json"

    plan = parse_command("Histogram of age", titanic, model_caller=invalid)
    result = execute_plan(validate_plan(plan, titanic), titanic)
    assert len(attempts) == 2
    assert isinstance(result.figure, Figure)
    plt.close(result.figure)


def test_apptest_renders_one_matplotlib_plot_for_each_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify every acceptance command reaches st.pyplot without app errors."""
    content = FIXTURE.read_bytes()
    rendered: list[Figure] = []
    real_pyplot = app.st.pyplot

    def capture_plot(figure: Figure | None = None, *args: object, **kwargs: object) -> None:
        if isinstance(figure, Figure):
            rendered.append(figure)
        real_pyplot(figure, *args, **kwargs)

    monkeypatch.setattr(app.st, "pyplot", capture_plot)
    monkeypatch.setattr(
        "src.command_parser._invoke_model",
        lambda *_: (_ for _ in ()).throw(RuntimeError("offline AppTest")),
    )
    monkeypatch.setenv("AI_PROVIDER", "offline")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    test_app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    test_app.run()
    test_app.file_uploader[0].set_value(
        ("titanic_sample.csv", content, "text/csv")
    ).run()
    assert not test_app.exception

    commands = [item[0] for item in ACCEPTANCE] + NEW_ACCEPTANCE
    for command in commands:
        start = len(rendered)
        test_app.text_input(key="ask_command").set_value(command)
        test_app.button(key="ask-run").click().run()
        assert not test_app.exception, f"{command}: {test_app.exception}"
        result_figure = test_app.session_state["ask_result"]["figure"]
        assert isinstance(result_figure, Figure)
        assert test_app.session_state["ask_result"]["message"].startswith("I understood:")
        assert any(figure is result_figure for figure in rendered[start:])
        assert result_figure._suptitle is not None or any(
            axis.get_title() for axis in result_figure.axes
        )

    test_app.text_input(key="ask_command").set_value("plot unknown metric")
    test_app.button(key="ask-run").click().run()
    assert not test_app.exception
    assert isinstance(test_app.session_state["ask_result"]["figure"], Figure)

    skewed_csv = pd.DataFrame({
        "Fare": [float(value) for value in range(1, 12)] + [1000.0],
    }).to_csv(index=False).encode("utf-8")
    test_app.file_uploader[0].set_value(
        ("skewed.csv", skewed_csv, "text/csv")
    ).run()
    test_app.text_input(key="ask_command").set_value("histogram of fare")
    test_app.button(key="ask-run").click().run()
    assert not test_app.exception
    assert test_app.checkbox(key="ask-log-scale").value is False
    test_app.checkbox(key="ask-log-scale").set_value(True).run()
    assert not test_app.exception
    assert test_app.session_state["ask_result"]["figure"].axes[0].get_yscale() == "log"
