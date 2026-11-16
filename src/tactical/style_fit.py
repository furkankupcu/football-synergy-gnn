"""Per-player style fit — the player's individual metrics ↔ the target style fingerprint.

Approach:
1. Target style fingerprint (team-level, percentile 0-100): `STYLE_TEMPLATES` or a reference team.
2. The player's position-specific KEY metrics are converted to league percentiles.
3. For each metric the "should be high / should be low" direction comes from the style target
   (above 50 → should be high, below → should be low). The more the player's percentile
   follows that direction, the more points it earns.
4. Total fit % = mean metric fit (0-100). Per-metric attribution: which metrics
   contributed most (direction + percentile) and which held the score back.

This module is a pragmatic way to reduce a team-level style frame to a per-player score.
It is not pure — for a PSG-style target, "high tackles" does not generalise to all 11
players (CB ↔ ST differ). The position-specific metric map mostly solves this.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


# Position → the per-90 metrics that "define this player's style".
# Mapped to table columns (`player_stats`).
# Each metric: (column_name, "high_good" = True / "low_good" = False)
# The style target is team-level, so we collect this metric set for the player;
# the direction comes from the style fingerprint (see below).
POS_METRICS: dict[str, list[tuple[str, str]]] = {
    "GK":  [("saves", "high"), ("punches", "high"), ("good_high_claim", "high"),
            ("accurate_pass", "high"), ("total_long_balls", "neutral"),
            ("accurate_long_balls", "high")],
    "RB":  [("tackles", "high"), ("interceptions", "high"), ("accurate_cross", "high"),
            ("total_cross", "neutral"), ("won_contest", "high"), ("key_pass", "high"),
            ("accurate_pass", "high"), ("duel_won", "high"), ("touches", "neutral")],
    "LB":  [("tackles", "high"), ("interceptions", "high"), ("accurate_cross", "high"),
            ("total_cross", "neutral"), ("won_contest", "high"), ("key_pass", "high"),
            ("accurate_pass", "high"), ("duel_won", "high"), ("touches", "neutral")],
    "RCB": [("interceptions", "high"), ("blocks", "high"), ("clearance", "high"),
            ("aerial_won", "high"), ("accurate_pass", "high"), ("accurate_long_balls", "high"),
            ("duel_won", "high"), ("ground_duels_won", "high")],
    "LCB": [("interceptions", "high"), ("blocks", "high"), ("clearance", "high"),
            ("aerial_won", "high"), ("accurate_pass", "high"), ("accurate_long_balls", "high"),
            ("duel_won", "high"), ("ground_duels_won", "high")],
    "CDM": [("interceptions", "high"), ("tackles", "high"), ("accurate_pass", "high"),
            ("key_pass", "high"), ("duel_won", "high"), ("ball_recovery", "high"),
            ("total_long_balls", "neutral"), ("accurate_long_balls", "high")],
    "CM":  [("accurate_pass", "high"), ("key_pass", "high"), ("big_chance_created", "high"),
            ("interceptions", "high"), ("tackles", "high"), ("duel_won", "high"),
            ("expected_assists", "high"), ("touches", "neutral")],
    "CAM": [("key_pass", "high"), ("big_chance_created", "high"), ("expected_assists", "high"),
            ("goal_assist", "high"), ("accurate_pass", "high"), ("won_contest", "high"),
            ("expected_goals", "high"), ("total_shots", "neutral")],
    "RW":  [("won_contest", "high"), ("accurate_cross", "high"), ("key_pass", "high"),
            ("expected_assists", "high"), ("expected_goals", "high"),
            ("on_target_scoring_attempt", "high"), ("total_shots", "neutral"),
            ("goals", "high")],
    "LW":  [("won_contest", "high"), ("accurate_cross", "high"), ("key_pass", "high"),
            ("expected_assists", "high"), ("expected_goals", "high"),
            ("on_target_scoring_attempt", "high"), ("total_shots", "neutral"),
            ("goals", "high")],
    "ST":  [("expected_goals", "high"), ("goals", "high"), ("on_target_scoring_attempt", "high"),
            ("total_shots", "high"), ("aerial_won", "high"), ("big_chance_created", "neutral"),
            ("key_pass", "neutral"), ("ball_recovery", "high")],
}

# Style-target team metrics → player-metric family for the "this metric should be
# high" direction. A player's style fit is measured by whether the IN-ROLE percentile
# of their position's key metrics follows the direction the style target asks for in
# THIS KIND of metrics.
STYLE_TO_PLAYER_FAMILY: dict[str, list[str]] = {
    # attacking volume
    "shots": ["total_shots", "on_target_scoring_attempt", "expected_goals", "goals"],
    "shots_on_target": ["on_target_scoring_attempt", "goals"],
    "xg": ["expected_goals"],
    "goals_for": ["goals", "expected_goals"],
    # creativity
    "progressive_passes": ["accurate_pass", "key_pass", "accurate_long_balls"],
    "progressive_carries": ["won_contest", "touches"],
    "passes_into_final_third": ["accurate_pass", "key_pass"],
    "passes_into_penalty_area": ["key_pass", "big_chance_created"],
    # top hakimiyeti / pas kalitesi
    "possession": ["touches", "accurate_pass"],
    "passes_pct": ["accurate_pass"],
    # defansif aksiyon
    "tackles": ["tackles", "ground_duels_won"],
    "interceptions": ["interceptions", "ball_recovery"],
    "blocks": ["blocks", "clearance"],
    # 1v1 / dribling
    "take_ons_attempted": ["won_contest"],
    "take_ons_won": ["won_contest"],
    # savunma sonucu
    "goals_against": ["clearance", "blocks", "interceptions"],
    "xg_against": ["clearance", "blocks", "interceptions"],
    "average_shot_distance": [],  # weak player-level equivalent, skip
}


def _per90(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert the given columns to per-90. Leaves NaN where minutes_played is 0."""
    out = df.copy()
    minutes = pd.to_numeric(out.get("minutes_played", 0), errors="coerce").fillna(0)
    safe = minutes.where(minutes > 0)
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") / safe * 90.0
    return out


def aggregate_player_metrics(
    player_stats: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """Aggregate match-level rows per player: per-90 average.

    Input: `player_stats` benzeri (match_id, source_player_id, *metric_cols, minutes_played).
    Output: index=source_player_id, columns=metrics + ['minutes', 'matches'].
    """
    df = player_stats.copy()
    df["minutes_played"] = pd.to_numeric(df.get("minutes_played", 0), errors="coerce").fillna(0)
    agg_cols = [c for c in metrics if c in df.columns]
    sums = df.groupby("source_player_id")[agg_cols].sum(numeric_only=True)
    mins = df.groupby("source_player_id")["minutes_played"].sum().rename("minutes")
    matches = df.groupby("source_player_id").size().rename("matches")
    result = sums.div(mins.replace(0, np.nan), axis=0) * 90.0
    result = result.join(mins).join(matches)
    return result


def style_fit_for_lineup(
    lineup: dict[str, str],            # {slot_pos → source_player_id}
    player_metrics: pd.DataFrame,      # per-90 metrikler, index=source_player_id
    style_vector: Optional[pd.Series], # team-level percentile target (style_templates)
    league_pool: pd.DataFrame,         # per-90 metrics of players in the same league (percentile base)
) -> dict:
    """Style fit score + metric attribution for every player.

    - league_pool: used for the percentile computation (position-filtered would be more
      accurate; global league for now).
    - If style_vector is None only a role-based fit % is returned ("high = good" for every metric).
    """
    per_player = []
    overall_scores = []

    for slot_pos, pid in lineup.items():
        # tolerate the ss_ prefix
        spid = int(str(pid).replace("ss_", "")) if str(pid).startswith("ss_") else int(pid)
        if spid not in player_metrics.index:
            per_player.append({"slot": slot_pos, "pid": f"ss_{spid}", "fit_pct": None,
                               "contributing": [], "note": "no metrics"})
            continue

        pos_metrics = POS_METRICS.get(slot_pos.upper(), POS_METRICS.get("CM"))
        contribs = []
        scores = []
        for col, direction in pos_metrics:
            if col not in player_metrics.columns:
                continue
            val = player_metrics.at[spid, col]
            if pd.isna(val):
                continue
            # league percentile
            pool_vals = league_pool[col].dropna() if col in league_pool.columns else pd.Series(dtype=float)
            if len(pool_vals) < 5:
                continue
            pct = float((pool_vals < val).mean())  # 0..1
            # style direction
            if style_vector is not None:
                # find the style family this metric belongs to
                aligned_dir = _direction_for_metric(col, style_vector)
            else:
                aligned_dir = "high" if direction != "low" else "low"
            # fit: direction + percentile combined
            if aligned_dir == "high":
                score = pct * 100
            elif aligned_dir == "low":
                score = (1 - pct) * 100
            else:
                # neutral → being near the middle percentile is fine (not bad) — fixed 75
                score = 70.0
            scores.append(score)
            contribs.append({
                "metric": col, "value": float(val),
                "percentile": round(pct * 100, 1),
                "direction": aligned_dir,
                "score": round(score, 1),
            })

        fit_pct = float(np.mean(scores)) if scores else None
        # contribs by impact: top 3 supports + bottom 2 weaknesses
        contribs_sorted = sorted(contribs, key=lambda x: -x["score"])
        top_pos = contribs_sorted[:3]
        bot = [c for c in reversed(contribs_sorted) if c["score"] < 50][:2]

        per_player.append({
            "slot": slot_pos, "pid": f"ss_{spid}",
            "fit_pct": round(fit_pct, 1) if fit_pct is not None else None,
            "contributing": top_pos,
            "weaknesses": bot,
        })
        if fit_pct is not None:
            overall_scores.append(fit_pct)

    team_fit = round(float(np.mean(overall_scores)), 1) if overall_scores else None
    return {"team_style_fit": team_fit, "per_player_fit": per_player}


def _direction_for_metric(player_col: str, style_vector: pd.Series) -> str:
    """Decide the style direction (high/low/neutral) for a player metric.

    Strategy: are there team-style families containing this player metric? If style_vector
    rates that family >55 → 'high', <45 → 'low', in between → 'neutral'.
    """
    # reverse mapping: player_col → which style metrics "cover" this metric
    style_keys = [sk for sk, members in STYLE_TO_PLAYER_FAMILY.items() if player_col in members]
    if not style_keys:
        return "high"  # unknown → default high-is-good
    vals = [float(style_vector.get(sk, 50)) for sk in style_keys]
    avg = float(np.mean(vals))
    if avg >= 60:
        return "high"
    if avg <= 40:
        return "low"
    return "neutral"
