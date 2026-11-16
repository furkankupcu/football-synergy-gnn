"""A club's XI in a season: the most-used one, or the one with the highest predicted synergy.

most_used_xi: goalkeeper with most starts + the ten outfield players with most starts (ties by
minutes), formation-agnostic. Each player gets his season-average position and the role derived
from it with the same rule the graph uses (`coordinate_to_position`), so a new signing shows the
role he plays now, not the one at his previous club.

best_synergy_xi: keeps the team's own shape (the roles of the most-used XI) and fills it with
squad players so that the summed pairwise predicted synergy is maximal.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from src import db
from src.graph.edge_builder import _season_where
from src.graph.positions import FORMATIONS, coordinate_to_position

# Which squad roles may fill a slot of the team's shape (inverted wingers, 6/8 swaps, CB pairs)
COMPATIBLE = {
    "GK": {"GK"}, "RB": {"RB", "RWB"}, "LB": {"LB", "LWB"}, "RWB": {"RWB", "RB"}, "LWB": {"LWB", "LB"},
    "RCB": {"RCB", "LCB"}, "LCB": {"LCB", "RCB"},
    # CAM also takes wide players: inside forwards play as a 10, and left/right-sided 8s get a
    # wing label from the coordinate rule (fluid sides such as PSG need this)
    "CDM": {"CDM", "CM"}, "CM": {"CM", "CDM", "CAM"}, "CAM": {"CAM", "CM", "LW", "RW"},
    "RW": {"RW", "LW"}, "LW": {"LW", "RW"}, "ST": {"ST"},
}


def squad(team: str, season: str) -> pd.DataFrame:
    """Everyone who played for the team in the season. Columns: player_id ('ss_<id>'), name,
    starts, minutes, x, y (0–100, x towards the opponent goal, y 0 = right touchline; NaN without
    position data), role."""
    where_sql, params = _season_where(season, None)
    extra = "AND" if where_sql else "WHERE"
    df = db.read_sql(f"""
        SELECT 'ss_' || p.source_player_id::text AS player_id,
               MAX(p.player_name)                     AS name,
               SUM(CASE WHEN p.substitute IS FALSE THEN 1 ELSE 0 END) AS starts,
               COALESCE(SUM(p.minutes_played), 0)     AS minutes,
               MODE() WITHIN GROUP (ORDER BY p.position) AS base,
               -- avg position weighted by point count; rows without a count weigh 1
               SUM(a.avg_x * COALESCE(a.points_count, 1)) / NULLIF(SUM(CASE WHEN a.avg_x IS NOT NULL
                   THEN COALESCE(a.points_count, 1) END), 0) AS x,
               SUM(a.avg_y * COALESCE(a.points_count, 1)) / NULLIF(SUM(CASE WHEN a.avg_y IS NOT NULL
                   THEN COALESCE(a.points_count, 1) END), 0) AS y
        FROM player_stats p
        JOIN matches m ON m.match_id = p.match_id
        LEFT JOIN player_avg_positions a
               ON a.match_id = p.match_id AND a.source_player_id = p.source_player_id
        {where_sql} {extra} p.team_name = :team
        GROUP BY p.source_player_id
    """, {**params, "team": team})
    if df.empty:
        raise ValueError(f"{team}: no matches in {season}")
    df["role"] = [
        "GK" if b == "G" else coordinate_to_position(x, y, b) if pd.notna(x) and pd.notna(y)
        else coordinate_to_position(None, None, b)
        for b, x, y in zip(df["base"], df["x"], df["y"])
    ]
    return df.drop(columns="base").sort_values(["starts", "minutes"], ascending=False,
                                               ignore_index=True)


def most_used_xi(team: str, season: str, include: list[str] | None = None,
                 exclude: list[str] | None = None) -> pd.DataFrame:
    """Same columns as `squad`, 11 rows, GK first.

    include / exclude: name fragments (case-insensitive) to force a player in or out — data
    alone cannot know that e.g. a first-choice player missed games through injury.
    """
    df = squad(team, season)
    match = lambda frags: df["name"].str.lower().apply(
        lambda n: any(f.lower() in n for f in frags or []))
    if exclude:
        df = df[~match(exclude)]
    missing = [f for f in include or [] if not df["name"].str.lower().str.contains(f.lower()).any()]
    if missing:
        raise ValueError(f"{team} {season}: no player matching {missing}")
    df = df.assign(_forced=match(include)).sort_values(["_forced", "starts", "minutes"],
                                                         ascending=False)
    gk = df[df["role"] == "GK"].head(1)
    return pd.concat([gk, df[df["role"] != "GK"].head(10)], ignore_index=True).drop(columns="_forced")


def best_synergy_xi(eng, team: str, season: str, start: pd.DataFrame,
                    min_minutes: int = 90, formation: str | None = None
                    ) -> tuple[pd.DataFrame, float, float]:
    """Hill-climb: replace players while the summed pairwise predicted synergy of the XI rises
    (attack + defence, each in units of its spread, when the defensive model exists).
    Slots = the roles of `start` (a most-used XI), or of `formation` (a FORMATIONS key) when
    given; a squad player may fill a slot if his role is COMPATIBLE. Without a formation,
    players unknown to the model stay where they are (no prediction possible).
    Returns (xi with a `slot` column, score of `start`, best score)."""
    pool = squad(team, season)
    pool = pool[(pool["minutes"] >= min_minutes) | pool["player_id"].isin(start["player_id"])]
    pool = pool[pool["player_id"].isin(eng.pid_to_idx)].reset_index(drop=True)
    idx = [eng.pid_to_idx[p] for p in pool["player_id"]]
    a, b = zip(*combinations(range(len(idx)), 2))
    pa, pb = [idx[i] for i in a], [idx[i] for i in b]
    # Attack (+ defence if that model exists), each scaled by its own spread so neither dominates
    kinds = ["attack", "defence"] if eng.def_model is not None else ["attack"]
    s = sum((v := eng.pair_scores(pa, pb, k).numpy()) / (v.std() or 1.0) for k in kinds)
    S = np.zeros((len(idx), len(idx)))
    S[a, b] = s; S[b, a] = s

    row_of = {p: k for k, p in enumerate(pool["player_id"])}
    fits = lambda r, slot: pool.at[r, "role"] in COMPATIBLE.get(slot, {slot})

    def score(lu):
        known = [r for r in lu if r is not None]
        return float(sum(S[i, j] for i, j in combinations(known, 2)))

    start_lineup = [row_of.get(p) for p in start["player_id"]]   # None = unknown player, fixed
    start_score = score(start_lineup)
    if formation:
        # A heavily rotated side's most-used XI is no real shape (e.g. three CBs, no RB):
        # take the slots from the formation and seed each with its most-used compatible player.
        slots = list(FORMATIONS[formation])
        lineup: list[int | None] = []
        for slot in slots:
            if slot == "GK":   # keep the most-used goalkeeper
                lineup.append(start_lineup[list(start["role"]).index("GK")])
                continue
            cand = [r for r in range(len(pool)) if r not in lineup and fits(r, slot)]
            if not cand:
                raise ValueError(f"{team}: no squad player fits {slot} in {formation}")
            lineup.append(cand[0])   # pool is sorted by starts, minutes
        fixed = {slots.index("GK")}
    else:
        slots = list(start["role"])                       # the team's shape
        lineup = start_lineup
        fixed = {k for k, r in enumerate(lineup) if r is None or slots[k] == "GK"}

    best = score(lineup)
    improved = True
    while improved:
        improved = False
        for k in range(len(slots)):
            if k in fixed:
                continue
            # replace with a bench player (the score depends on the set of players only,
            # so swapping two XI players between slots never changes it)
            for r in range(len(pool)):
                if r in lineup or not fits(r, slots[k]):
                    continue
                trial = lineup.copy(); trial[k] = r
                if (v := score(trial)) > best + 1e-9:
                    lineup, best, improved = trial, v, True

    out = pd.concat([pool.loc[[r]] if r is not None else start.loc[[k]]
                     for k, r in enumerate(lineup)], ignore_index=True)
    assert out["player_id"].is_unique and len(out) == len(slots)
    out["slot"] = slots
    return out, start_score, best
