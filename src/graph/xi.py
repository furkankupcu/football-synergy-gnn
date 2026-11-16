"""A club's most-used XI in a season, formation-agnostic.

The goalkeeper with most starts + the ten outfield players with most starts (ties broken by
minutes). Each player gets his season-average position and the role derived from it with the
same rule the graph uses (`coordinate_to_position`), so a new signing shows the role he plays
now, not the one at his previous club.
"""
from __future__ import annotations

import pandas as pd

from src import db
from src.graph.edge_builder import _season_where
from src.graph.positions import coordinate_to_position


def most_used_xi(team: str, season: str, include: list[str] | None = None,
                 exclude: list[str] | None = None) -> pd.DataFrame:
    """Columns: player_id ('ss_<id>'), name, starts, minutes, x, y (0–100, x towards the
    opponent goal, y 0 = right touchline; NaN without position data), role. GK first.

    include / exclude: name fragments (case-insensitive) to force a player in or out — data
    alone cannot know that e.g. a first-choice player missed games through injury.
    """
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
    match = lambda frags: df["name"].str.lower().apply(
        lambda n: any(f.lower() in n for f in frags or []))
    if exclude:
        df = df[~match(exclude)]
    df = df.assign(_forced=match(include)).sort_values(["_forced", "starts", "minutes"],
                                                         ascending=False)
    missing = [f for f in include or [] if not df["name"].str.lower().str.contains(f.lower()).any()]
    if missing:
        raise ValueError(f"{team} {season}: no player matching {missing}")
    gk = df[df["base"] == "G"].head(1)
    xi = pd.concat([gk, df[df["base"] != "G"].head(10)], ignore_index=True).drop(columns="_forced")
    xi["role"] = [
        "GK" if b == "G" else coordinate_to_position(x, y, b) if pd.notna(x) and pd.notna(y)
        else coordinate_to_position(None, None, b)
        for b, x, y in zip(xi["base"], xi["x"], xi["y"])
    ]
    return xi.drop(columns="base")
