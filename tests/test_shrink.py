import pandas as pd
import pytest

from src.graph.build import _shrink


def test_low_minutes_pulled_to_group_mean_regulars_barely_move():
    df = pd.DataFrame({
        "league": ["L"] * 3, "position": ["F"] * 3,
        "minutes": [3000.0, 3000.0, 20.0],
        "goals": [0.5, 0.3, 4.5],          # per 90; the cameo looks like a superstar
    })
    out = _shrink(df, ["league", "position"], ["goals"], k=450)
    prior = (0.5 * 3000 + 0.3 * 3000 + 4.5 * 20) / 6020
    assert out.loc[2, "goals"] == pytest.approx((20 * 4.5 + 450 * prior) / 470)
    assert out.loc[2, "goals"] < 0.7                       # no longer extreme
    assert abs(out.loc[0, "goals"] - 0.5) < 0.02           # regular almost unchanged


def test_zero_minutes_gets_the_prior():
    df = pd.DataFrame({"league": ["L", "L"], "position": ["D", "D"],
                       "minutes": [900.0, 0.0], "goals": [0.1, 0.0]})
    out = _shrink(df, ["league", "position"], ["goals"], k=450)
    assert out.loc[1, "goals"] == pytest.approx(0.1)
