import numpy as np
import pandas as pd
import pytest

from src.graph.edge_builder import (_aggregate, _per_match_pairs, _solo_defence, _team_conceded,
                                    _xg_between)

# Home team H: a plays 0-90, b plays 0-60, c comes on at 60. Away shots at 30 (0.2) and 75 (0.5).
LINEUP = pd.DataFrame({
    "match_id": ["m1"] * 3, "team_id": ["H"] * 3, "home_team_id": ["H"] * 3,
    "player_id": ["a", "b", "c"], "minutes_on": [0, 0, 60], "minutes_off": [90, 60, 90],
})
SHOTS = {("m1", False): (np.array([30.0, 75.0]), np.array([0.2, 0.5])),   # away shots
         ("m1", True): (np.array([10.0]), np.array([0.9]))}                # home shot: not conceded


def test_window_is_half_open():
    shots = (np.array([30.0, 60.0, 61.0]), np.array([1.0, 2.0, 4.0]))
    assert _xg_between(shots, 30, 60) == 2.0       # (30, 60]
    assert _xg_between(None, 0, 90) == 0.0


def test_pairs_count_only_shots_against_while_both_on():
    conc = _team_conceded(LINEUP, SHOTS)
    pairs = _per_match_pairs(LINEUP, "teammate", conc).set_index(["player_a", "player_b"])
    assert pairs.loc[("a", "b"), "xga_together"] == pytest.approx(0.2)   # 0-60
    assert pairs.loc[("a", "c"), "xga_together"] == pytest.approx(0.5)   # 60-90
    assert pairs.loc[("a", "b"), "def_minutes"] == 60


def test_match_without_shot_data_is_ignored():
    assert _team_conceded(LINEUP, {}) == {}
    pairs = _per_match_pairs(LINEUP, "teammate", {})
    assert (pairs["def_minutes"] == 0).all()


def test_solo_and_uplift_sign():
    solo = _solo_defence(LINEUP, SHOTS)
    assert solo["a"] == pytest.approx(0.7)          # 0.7 xG in 90 min
    assert solo["b"] == pytest.approx(0.2 / 60 * 90)
    pairs = _per_match_pairs(LINEUP, "teammate", _team_conceded(LINEUP, SHOTS)).assign(match_id="m1")
    pairs["minutes_together"] *= 2; pairs["def_minutes"] *= 2; pairs["xga_together"] *= 2  # ≥ 90 min
    agg = _aggregate(pairs, None, {}, solo).set_index(["player_a", "player_b"])
    # a+b concede 0.3/90 together vs expected (0.7 + 0.3) / 2 = 0.5 → positive (good)
    assert agg.loc[("a", "b"), "def_uplift"] == pytest.approx(0.5 - 0.3)
