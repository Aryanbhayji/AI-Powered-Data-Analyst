"""Regression tests for open-ended relationship questions."""

from pathlib import Path

import pandas as pd

from src.command_parser import rule_based_plan

TITANIC_FIXTURE = Path(__file__).parent / "fixtures" / "titanic_sample.csv"


def test_titanic_relationship_request_is_interpreted() -> None:
    """Reproduce the reported fare-versus-survival dead end."""
    dataframe = pd.read_csv(TITANIC_FIXTURE)

    plan = rule_based_plan(
        "Relationship between ticket fare and survival", dataframe
    )

    assert plan["action"] == "relationship"
    assert plan["var_a"] == "Fare"
    assert plan["var_b"] == "Survived"
