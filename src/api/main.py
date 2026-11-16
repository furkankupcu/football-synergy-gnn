"""FastAPI — match data API.

To run:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional
from datetime import date, datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src import db


def _to_python(v: Any) -> Any:
    """Convert numpy / pandas / datetime types into JSON-serialisable Python types."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v

import json as _json_mod


class _SafeJSONResponse(JSONResponse):
    """JSONResponse that safely serialises numpy/date types."""

    def render(self, content: Any) -> bytes:
        class _Enc(_json_mod.JSONEncoder):
            def default(self, o):
                return _to_python(o)

        return _json_mod.dumps(content, cls=_Enc, ensure_ascii=False).encode("utf-8")


app = FastAPI(
    title="Footscrapping API",
    description="match, player, shot and team statistics",
    version="1.0.0",
    default_response_class=_SafeJSONResponse,
)

_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.api.gnn_routes import router as _gnn_router
app.include_router(_gnn_router)
from src.api.tactical_routes import router as _tactical_router
app.include_router(_tactical_router)


def _df_to_list(df: pd.DataFrame) -> list[dict]:
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    return [{k: _to_python(v) for k, v in row.items()} for row in records]



# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(_STATIC / "index.html")


@app.get("/health", tags=["info"])
def health():
    try:
        tables = db.list_tables()
        ss_tables = tables
        return {"status": "ok", "tables": ss_tables}
    except Exception as e:
        return {"status": "db_error", "detail": str(e)}


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------

@app.get("/matches", tags=["matches"])
def list_matches(
    league: Optional[str] = Query(None, description="e.g. 'Turkiye Super Lig'"),
    season: Optional[str] = Query(None, description="e.g. '25/26'"),
    round: Optional[int] = Query(None),
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None),
    team: Optional[str] = Query(None, description="Home or away team name (partial match)"),
    has_xg: Optional[bool] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    filters = []
    params: dict = {}

    if league:
        filters.append("league = :league")
        params["league"] = league
    if season:
        filters.append("season = :season")
        params["season"] = season
    if round is not None:
        filters.append("round = :round")
        params["round"] = round
    if date_from:
        filters.append("match_date >= :date_from")
        params["date_from"] = str(date_from)
    if date_to:
        filters.append("match_date <= :date_to")
        params["date_to"] = str(date_to)
    if team:
        filters.append("(home_team_name ILIKE :team OR away_team_name ILIKE :team)")
        params["team"] = f"%{team}%"
    if has_xg is not None:
        filters.append("has_xg = :has_xg")
        params["has_xg"] = has_xg

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    params["limit"] = limit
    params["offset"] = offset

    query = f"""
        SELECT match_id, source_id, league, season, match_date, round,
               home_team_name, away_team_name,
               home_goals, away_goals, home_goals_ht, away_goals_ht,
               winner_code, has_xg
        FROM matches
        {where}
        ORDER BY match_date DESC, match_id
        LIMIT :limit OFFSET :offset
    """
    try:
        df = db.read_sql(query, params)
        return {"total": len(df), "offset": offset, "data": _df_to_list(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/matches/{match_id}", tags=["matches"])
def get_match(match_id: str):
    try:
        df = db.read_sql(
            "SELECT * FROM matches WHERE match_id = :mid",
            {"mid": match_id},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if df.empty:
        raise HTTPException(status_code=404, detail="Match not found")
    return _df_to_list(df)[0]


@app.get("/matches/{match_id}/player-stats", tags=["matches"])
def match_player_stats(match_id: str, team_id: Optional[int] = None):
    filters = ["match_id = :mid"]
    params: dict = {"mid": match_id}
    if team_id is not None:
        filters.append("team_id = :team_id")
        params["team_id"] = team_id
    where = "WHERE " + " AND ".join(filters)
    try:
        df = db.read_sql(
            f"SELECT * FROM player_stats {where} ORDER BY team_name, minutes_played DESC",
            params,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"match_id": match_id, "count": len(df), "data": _df_to_list(df)}


@app.get("/matches/{match_id}/shots", tags=["matches"])
def match_shots(match_id: str, is_home: Optional[bool] = None):
    filters = ["match_id = :mid"]
    params: dict = {"mid": match_id}
    if is_home is not None:
        filters.append("is_home = :is_home")
        params["is_home"] = is_home
    where = "WHERE " + " AND ".join(filters)
    try:
        df = db.read_sql(
            f"SELECT * FROM shots {where} ORDER BY time_minute, time_seconds",
            params,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"match_id": match_id, "count": len(df), "data": _df_to_list(df)}


@app.get("/matches/{match_id}/team-stats", tags=["matches"])
def match_team_stats(match_id: str, period: Optional[str] = Query(None, description="ALL | 1ST | 2ND")):
    filters = ["match_id = :mid"]
    params: dict = {"mid": match_id}
    if period:
        filters.append("period = :period")
        params["period"] = period.upper()
    where = "WHERE " + " AND ".join(filters)
    try:
        df = db.read_sql(
            f"SELECT * FROM team_stats {where} ORDER BY period, team_side",
            params,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"match_id": match_id, "count": len(df), "data": _df_to_list(df)}


# ---------------------------------------------------------------------------
# Oyuncular
# ---------------------------------------------------------------------------

@app.get("/players", tags=["players"])
def list_players(
    name: Optional[str] = Query(None, description="Player name (partial match)"),
    team_name: Optional[str] = Query(None),
    position: Optional[str] = Query(None, description="Position (GK/RB/LB/RCB/LCB/CDM/CM/CAM/RW/LW/ST) or base G/D/M/F"),
    limit: int = Query(50, ge=1, le=200),
):
    filters = []
    params: dict = {}

    pos_upper = position.upper() if position else None
    detailed_pos = pos_upper if pos_upper in _POS_REQUIRED else None
    # Detailed position → pre-filter by base codes in SQL (narrows the pool and keeps
    # ORDER BY total_goals from truncating in favour of forwards)
    _DETAIL_TO_BASES = {
        "GK":  ["G"],
        "RB":  ["D"], "LB":  ["D"], "RCB": ["D"], "LCB": ["D"],
        "CDM": ["M", "D"], "CM": ["M"], "CAM": ["M", "F"],
        "RW":  ["F", "M"], "LW": ["F", "M"], "ST": ["F"],
    }

    if name:
        filters.append("player_name ILIKE :name")
        params["name"] = f"%{name}%"
    if team_name:
        filters.append("team_name ILIKE :team_name")
        params["team_name"] = f"%{team_name}%"
    if pos_upper and not detailed_pos:
        filters.append("position = :position")
        params["position"] = pos_upper
    if detailed_pos:
        bases = _DETAIL_TO_BASES.get(detailed_pos, [])
        if bases:
            placeholders = ", ".join(f":pb{i}" for i in range(len(bases)))
            filters.append(f"position IN ({placeholders})")
            for i, b in enumerate(bases):
                params[f"pb{i}"] = b

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    # The detailed position filter is applied later in Python → keep the SQL wide
    params["limit"] = max(limit * 10, 200) if detailed_pos else limit

    query = f"""
        SELECT
            source_player_id,
            player_name,
            team_name,
            position,
            COUNT(DISTINCT match_id) AS matches_played,
            ROUND(AVG(rating)::numeric, 2) AS avg_rating,
            SUM(goals) AS total_goals,
            SUM(goal_assist) AS total_assists,
            ROUND(SUM(expected_goals)::numeric, 2) AS total_xg,
            ROUND(SUM(expected_assists)::numeric, 2) AS total_xa,
            SUM(minutes_played) AS total_minutes
        FROM player_stats
        {where}
        GROUP BY source_player_id, player_name, team_name, position
        ORDER BY total_goals DESC NULLS LAST, avg_rating DESC NULLS LAST
        LIMIT :limit
    """
    try:
        df = db.read_sql(query, params)
        if detailed_pos and not df.empty:
            df = apply_resolved_position(df, id_col="source_player_id",
                                         pos_col="position", season=None)
            df = df[df["position"].str.upper() == detailed_pos].head(limit)
        return {"count": len(df), "data": _df_to_list(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/players/{player_id}", tags=["players"])
def get_player(player_id: int):
    try:
        summary_df = db.read_sql(
            """
            SELECT
                source_player_id, player_name, team_name, position,
                COUNT(DISTINCT match_id) AS matches_played,
                ROUND(AVG(rating)::numeric, 2) AS avg_rating,
                SUM(goals) AS total_goals,
                SUM(goal_assist) AS total_assists,
                ROUND(SUM(expected_goals)::numeric, 2) AS total_xg,
                ROUND(SUM(expected_assists)::numeric, 2) AS total_xa,
                SUM(minutes_played) AS total_minutes,
                SUM(total_shots) AS total_shots,
                SUM(total_pass) AS total_passes,
                ROUND(AVG(accurate_pass::float / NULLIF(total_pass, 0) * 100)::numeric, 1) AS pass_accuracy_pct,
                SUM(duel_won) AS total_duel_won,
                SUM(duel_lost) AS total_duel_lost
            FROM player_stats
            WHERE source_player_id = :pid
            GROUP BY source_player_id, player_name, team_name, position
            """,
            {"pid": player_id},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if summary_df.empty:
        raise HTTPException(status_code=404, detail="Player not found")

    try:
        matches_df = db.read_sql(
            """
            SELECT ps.match_id, m.match_date, m.home_team_name, m.away_team_name,
                   m.home_goals, m.away_goals,
                   ps.team_name, ps.position, ps.minutes_played,
                   ps.rating, ps.goals, ps.goal_assist,
                   ps.expected_goals, ps.expected_assists, ps.total_shots
            FROM player_stats ps
            JOIN matches m ON m.match_id = ps.match_id
            WHERE ps.source_player_id = :pid
            ORDER BY m.match_date DESC
            """,
            {"pid": player_id},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "summary": _df_to_list(summary_df)[0],
        "matches": _df_to_list(matches_df),
    }


@app.get("/players/{player_id}/shots", tags=["players"])
def player_shots(player_id: int):
    try:
        name_df = db.read_sql(
            "SELECT DISTINCT player_name FROM player_stats WHERE source_player_id = :pid LIMIT 1",
            {"pid": player_id},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if name_df.empty:
        raise HTTPException(status_code=404, detail="Player not found")

    player_name = name_df.iloc[0]["player_name"]
    try:
        df = db.read_sql(
            "SELECT * FROM shots WHERE player_name = :name ORDER BY match_id, time_minute",
            {"name": player_name},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"player_name": player_name, "count": len(df), "data": _df_to_list(df)}


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

@app.get("/teams", tags=["teams"])
def list_teams(league: Optional[str] = None, season: Optional[str] = None):
    filters = []
    params: dict = {}
    if league:
        filters.append("m.league = :league")
        params["league"] = league
    if season:
        filters.append("m.season = :season")
        params["season"] = season
    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    query = f"""
        SELECT team_id, team_name, COUNT(DISTINCT match_id) AS matches
        FROM (
            SELECT home_team_id AS team_id, home_team_name AS team_name, match_id
            FROM matches m {where}
            UNION ALL
            SELECT away_team_id, away_team_name, match_id
            FROM matches m {where}
        ) t
        GROUP BY team_id, team_name
        ORDER BY team_name
    """
    try:
        df = db.read_sql(query, params)
        return {"count": len(df), "data": _df_to_list(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/teams/{team_id}", tags=["teams"])
def get_team(team_id: int, season: Optional[str] = None):
    params: dict = {"tid": team_id}
    season_filter = ""
    if season:
        season_filter = "AND season = :season"
        params["season"] = season

    try:
        matches_df = db.read_sql(
            f"""
            SELECT match_id, match_date, league, season, round,
                   home_team_name, away_team_name,
                   home_goals, away_goals, winner_code,
                   CASE WHEN home_team_id = :tid THEN 'home' ELSE 'away' END AS side
            FROM matches
            WHERE (home_team_id = :tid OR away_team_id = :tid) {season_filter}
            ORDER BY match_date DESC
            """,
            params,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if matches_df.empty:
        raise HTTPException(status_code=404, detail="Team not found")

    try:
        stats_df = db.read_sql(
            f"""
            SELECT
                ts.*,
                m.home_team_name, m.away_team_name, m.match_date
            FROM team_stats ts
            JOIN matches m ON m.match_id = ts.match_id
            WHERE (m.home_team_id = :tid AND ts.team_side = 'home')
               OR (m.away_team_id = :tid AND ts.team_side = 'away')
            {season_filter.replace('season', 'm.season')}
            AND ts.period = 'ALL'
            ORDER BY m.match_date DESC
            """,
            params,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    team_name = matches_df.iloc[0]["home_team_name"] if matches_df.iloc[0]["side"] == "home" else matches_df.iloc[0]["away_team_name"]

    return {
        "team_id": team_id,
        "team_name": team_name,
        "matches": _df_to_list(matches_df),
        "team_stats": _df_to_list(stats_df),
    }



# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

@app.get("/stats/summary", tags=["stats"])
def summary(league: Optional[str] = None, season: Optional[str] = None):
    filters = []
    params: dict = {}
    if league:
        filters.append("league = :league")
        params["league"] = league
    if season:
        filters.append("season = :season")
        params["season"] = season
    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    try:
        match_df = db.read_sql(
            f"SELECT COUNT(*) AS total_matches, COUNT(DISTINCT league) AS leagues FROM matches {where}",
            params,
        )
        player_df = db.read_sql(
            f"""
            SELECT COUNT(*) AS total_player_rows,
                   COUNT(DISTINCT source_player_id) AS unique_players
            FROM player_stats ps
            JOIN matches m ON m.match_id = ps.match_id
            {where.replace('WHERE', 'WHERE') if where else ''}
            """,
            params,
        )
        shot_df = db.read_sql(
            f"""
            SELECT COUNT(*) AS total_shots,
                   ROUND(AVG(xg)::numeric, 4) AS avg_xg_per_shot
            FROM shots s
            JOIN matches m ON m.match_id = s.match_id
            {where.replace('WHERE', 'WHERE') if where else ''}
            """,
            params,
        )
        return {
            "matches": _df_to_list(match_df)[0],
            "players": _df_to_list(player_df)[0],
            "shots": _df_to_list(shot_df)[0],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/top-scorers", tags=["stats"])
def top_scorers(
    league: Optional[str] = None,
    season: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
):
    filters = []
    params: dict = {"limit": limit}
    if league:
        filters.append("m.league = :league")
        params["league"] = league
    if season:
        filters.append("m.season = :season")
        params["season"] = season
    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    query = f"""
        SELECT
            ps.source_player_id, ps.player_name, ps.team_name, ps.position,
            SUM(ps.goals) AS goals,
            SUM(ps.goal_assist) AS assists,
            ROUND(SUM(ps.expected_goals)::numeric, 2) AS xg,
            COUNT(DISTINCT ps.match_id) AS matches
        FROM player_stats ps
        JOIN matches m ON m.match_id = ps.match_id
        {where}
        GROUP BY ps.source_player_id, ps.player_name, ps.team_name, ps.position
        HAVING SUM(ps.goals) > 0
        ORDER BY goals DESC, xg DESC
        LIMIT :limit
    """
    try:
        df = db.read_sql(query, params)
        return {"count": len(df), "data": _df_to_list(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/top-ratings", tags=["stats"])
def top_ratings(
    league: Optional[str] = None,
    season: Optional[str] = None,
    min_matches: int = Query(5, description="Minimum number of matches"),
    limit: int = Query(20, ge=1, le=100),
):
    filters = []
    params: dict = {"limit": limit, "min_matches": min_matches}
    if league:
        filters.append("m.league = :league")
        params["league"] = league
    if season:
        filters.append("m.season = :season")
        params["season"] = season
    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    query = f"""
        SELECT
            ps.source_player_id, ps.player_name, ps.team_name, ps.position,
            COUNT(DISTINCT ps.match_id) AS matches,
            ROUND(AVG(ps.rating)::numeric, 2) AS avg_rating,
            SUM(ps.goals) AS goals,
            SUM(ps.goal_assist) AS assists
        FROM player_stats ps
        JOIN matches m ON m.match_id = ps.match_id
        {where}
        GROUP BY ps.source_player_id, ps.player_name, ps.team_name, ps.position
        HAVING COUNT(DISTINCT ps.match_id) >= :min_matches AND AVG(ps.rating) IS NOT NULL
        ORDER BY avg_rating DESC
        LIMIT :limit
    """
    try:
        df = db.read_sql(query, params)
        return {"count": len(df), "data": _df_to_list(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Advanced stats helpers
# ---------------------------------------------------------------------------

def _per90(series: pd.Series, minutes: pd.Series) -> pd.Series:
    return series / minutes.clip(lower=1) * 90


def _safe_pct(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.replace(0, np.nan) * 100).round(1)


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Derived columns and per-90 conversion."""
    if df.empty:
        return df
    df = df.copy()
    mp = df["minutes_played"].clip(lower=1)

    def col(name):
        return df[name].fillna(0) if name in df.columns else pd.Series(0, index=df.index)

    df["_def_duels"]     = col("duel_won")  + col("duel_lost")
    df["_def_duel_pct"]  = _safe_pct(col("duel_won"),  df["_def_duels"])
    df["_aerials"]       = col("aerial_won") + col("aerial_lost")
    df["_aerial_pct"]    = _safe_pct(col("aerial_won"), df["_aerials"])
    df["_pass_acc_pct"]  = _safe_pct(col("accurate_pass"), col("total_pass"))
    df["_long_acc_pct"]  = _safe_pct(col("accurate_long_balls"), col("total_long_balls"))
    df["_prog_pass_pct"] = _safe_pct(col("total_progression"), col("total_pass"))
    df["_cross_acc_pct"] = _safe_pct(col("accurate_cross"), col("total_cross"))
    df["_dribbles"]      = col("won_contest") + col("challenge_lost")
    df["_dribble_pct"]   = _safe_pct(col("won_contest"), df["_dribbles"])
    df["_conv_pct"]      = _safe_pct(col("goals"), col("total_shots"))

    vol_cols = [
        "total_pass", "total_progression", "interception_won",
        "progressive_ball_carries_count", "expected_assists", "expected_goals",
        "total_shots", "key_pass", "total_cross", "outfielder_block",
        "was_fouled", "goals", "accurate_opposition_half_passes",
        "_def_duels", "_aerials", "_dribbles",
    ]
    for c in vol_cols:
        if c in df.columns:
            df[c] = _per90(df[c].fillna(0), mp)
    return df


# ---------------------------------------------------------------------------
# Puan durumu
# ---------------------------------------------------------------------------

@app.get("/stats/standings", tags=["stats"])
def standings(
    league: Optional[str] = None,
    season: Optional[str] = None,
):
    filters, params = [], {}
    if league:
        filters.append("league = :league"); params["league"] = league
    if season:
        filters.append("season = :season"); params["season"] = season
    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    query = f"""
        SELECT team_name,
               COUNT(*) AS played,
               SUM(CASE WHEN won  THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN draw THEN 1 ELSE 0 END) AS draws,
               SUM(CASE WHEN lose THEN 1 ELSE 0 END) AS losses,
               SUM(gf) AS gf, SUM(ga) AS ga,
               SUM(gf)-SUM(ga) AS gd,
               SUM(CASE WHEN won THEN 3 WHEN draw THEN 1 ELSE 0 END) AS pts
        FROM (
            SELECT home_team_name AS team_name, home_goals AS gf, away_goals AS ga,
                   home_goals>away_goals AS won, home_goals=away_goals AS draw, home_goals<away_goals AS lose
            FROM matches {where}
            UNION ALL
            SELECT away_team_name, away_goals, home_goals,
                   away_goals>home_goals, home_goals=away_goals, away_goals<home_goals
            FROM matches {where}
        ) t GROUP BY team_name ORDER BY pts DESC, gd DESC, gf DESC
    """
    try:
        df = db.read_sql(query, params)
        return {"count": len(df), "data": _df_to_list(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Team advanced stats (standings + tactics)
# ---------------------------------------------------------------------------

@app.get("/stats/team-advanced", tags=["stats"])
def team_advanced(
    league: Optional[str] = None,
    season: Optional[str] = None,
):
    filters, params = [], {}
    if league:
        filters.append("league = :league"); params["league"] = league
    if season:
        filters.append("season = :season"); params["season"] = season
    where_m = ("WHERE " + " AND ".join(filters)) if filters else ""
    ts_extra = ("AND " + " AND ".join(f"m.{f}" for f in filters)) if filters else ""

    standings_q = f"""
        SELECT team_name,
               COUNT(*) AS played,
               SUM(CASE WHEN won  THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN draw THEN 1 ELSE 0 END) AS draws,
               SUM(CASE WHEN lose THEN 1 ELSE 0 END) AS losses,
               SUM(gf) AS gf, SUM(ga) AS ga,
               SUM(gf)-SUM(ga) AS gd,
               SUM(CASE WHEN won THEN 3 WHEN draw THEN 1 ELSE 0 END) AS pts
        FROM (
            SELECT home_team_name AS team_name, home_goals AS gf, away_goals AS ga,
                   home_goals>away_goals AS won, home_goals=away_goals AS draw, home_goals<away_goals AS lose
            FROM matches {where_m}
            UNION ALL
            SELECT away_team_name, away_goals, home_goals,
                   away_goals>home_goals, home_goals=away_goals, away_goals<home_goals
            FROM matches {where_m}
        ) t GROUP BY team_name ORDER BY pts DESC, gd DESC
    """
    ts_q = f"""
        SELECT
            CASE WHEN ts.team_side='home' THEN m.home_team_name ELSE m.away_team_name END AS team_name,
            ROUND(AVG(ts.ball_possession)::numeric,1)     AS avg_possession,
            ROUND(AVG(ts.expected_goals)::numeric,2)      AS avg_xg,
            ROUND(AVG(ts.total_shots)::numeric,1)         AS avg_shots,
            ROUND(AVG(ts.shots_on_target)::numeric,1)     AS avg_shots_ot,
            ROUND(AVG(ts.total_passes)::numeric,1)        AS avg_passes,
            ROUND(AVG(ts.accurate_passes_pct)::numeric,1) AS avg_pass_acc,
            ROUND(AVG(ts.key_passes)::numeric,1)          AS avg_key_passes,
            ROUND(AVG(ts.total_tackles)::numeric,1)       AS avg_tackles,
            ROUND(AVG(ts.interceptions)::numeric,1)       AS avg_interceptions,
            ROUND(AVG(ts.corner_kicks)::numeric,1)        AS avg_corners,
            ROUND(AVG(ts.yellow_cards)::numeric,2)        AS avg_yellows,
            ROUND(AVG(ts.fouls)::numeric,1)               AS avg_fouls
        FROM team_stats ts
        JOIN matches m ON m.match_id = ts.match_id
        WHERE ts.period='ALL' {ts_extra}
        GROUP BY team_name
    """
    try:
        st_df = db.read_sql(standings_q, params)
        ts_df = db.read_sql(ts_q, params)
        merged = st_df.merge(ts_df, on="team_name", how="left") if not st_df.empty else st_df
        return {"count": len(merged), "data": _df_to_list(merged)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Team xG vs actual goals (scatter data)
# ---------------------------------------------------------------------------

@app.get("/stats/team-xg", tags=["stats"])
def team_xg_scatter(
    league: Optional[str] = None,
    season: Optional[str] = None,
):
    filters, params = [], {}
    if league:
        filters.append("m.league = :league"); params["league"] = league
    if season:
        filters.append("m.season = :season"); params["season"] = season
    extra = ("AND " + " AND ".join(filters)) if filters else ""

    query = f"""
        SELECT team, ROUND(SUM(xg)::numeric,2) AS xg, SUM(goals) AS goals
        FROM (
            SELECT m.home_team_name AS team, ts.expected_goals AS xg, m.home_goals AS goals
            FROM team_stats ts
            JOIN matches m ON m.match_id = ts.match_id
            WHERE ts.period='ALL' AND ts.team_side='home' {extra}
            UNION ALL
            SELECT m.away_team_name, ts.expected_goals, m.away_goals
            FROM team_stats ts
            JOIN matches m ON m.match_id = ts.match_id
            WHERE ts.period='ALL' AND ts.team_side='away' {extra}
        ) t
        GROUP BY team ORDER BY xg DESC
    """
    try:
        df = db.read_sql(query, params)
        if not df.empty:
            df["xg_diff"] = (df["goals"] - df["xg"]).round(2)
        return {"count": len(df), "data": _df_to_list(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Player advanced stats (all derived + per-90 columns)
# ---------------------------------------------------------------------------

@app.get("/stats/player-advanced", tags=["stats"])
def player_advanced(
    league: Optional[str] = None,
    season: Optional[str] = None,
    min_minutes: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=3000),
):
    filters, params = [], {}
    if league:
        filters.append("m.league = :league"); params["league"] = league
    if season:
        filters.append("m.season = :season"); params["season"] = season
    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    params["min_minutes"] = min_minutes
    params["limit"] = limit

    query = f"""
        SELECT
            ps.source_player_id, ps.player_name, ps.team_name, ps.position,
            SUM(ps.minutes_played)                               AS minutes_played,
            COUNT(DISTINCT ps.match_id)                          AS matches_played,
            ROUND(AVG(ps.rating)::numeric, 2)                    AS rating,
            SUM(ps.goals)                                        AS goals,
            SUM(ps.goal_assist)                                  AS goal_assist,
            SUM(ps.expected_goals)                               AS expected_goals,
            SUM(ps.expected_assists)                             AS expected_assists,
            SUM(ps.total_shots)                                  AS total_shots,
            SUM(ps.total_pass)                                   AS total_pass,
            SUM(ps.accurate_pass)                                AS accurate_pass,
            SUM(COALESCE(ps.total_long_balls, 0))                AS total_long_balls,
            SUM(COALESCE(ps.accurate_long_balls, 0))             AS accurate_long_balls,
            SUM(COALESCE(ps.total_cross, 0))                     AS total_cross,
            SUM(COALESCE(ps.accurate_cross, 0))                  AS accurate_cross,
            SUM(COALESCE(ps.key_pass, 0))                        AS key_pass,
            SUM(COALESCE(ps.total_progression, 0))               AS total_progression,
            SUM(COALESCE(ps.progressive_ball_carries_count, 0))  AS progressive_ball_carries_count,
            SUM(COALESCE(ps.duel_won, 0))                        AS duel_won,
            SUM(COALESCE(ps.duel_lost, 0))                       AS duel_lost,
            SUM(COALESCE(ps.aerial_won, 0))                      AS aerial_won,
            SUM(COALESCE(ps.aerial_lost, 0))                     AS aerial_lost,
            SUM(COALESCE(ps.interception_won, 0))                AS interception_won,
            SUM(COALESCE(ps.outfielder_block, 0))                AS outfielder_block,
            SUM(COALESCE(ps.won_contest, 0))                     AS won_contest,
            SUM(COALESCE(ps.challenge_lost, 0))                  AS challenge_lost,
            SUM(COALESCE(ps.was_fouled, 0))                      AS was_fouled,
            SUM(COALESCE(ps.accurate_opposition_half_passes, 0)) AS accurate_opposition_half_passes
        FROM player_stats ps
        JOIN matches m ON m.match_id = ps.match_id
        {where}
        GROUP BY ps.source_player_id, ps.player_name, ps.team_name, ps.position
        HAVING SUM(ps.minutes_played) >= :min_minutes
        ORDER BY AVG(ps.rating) DESC NULLS LAST
        LIMIT :limit
    """
    try:
        df = db.read_sql(query, params)
        df = _add_derived(df)
        return {"count": len(df), "data": _df_to_list(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Rapor endpoints
# ---------------------------------------------------------------------------

import io as _io
import base64 as _b64
import os as _os

def _stream_pdf(path: str, filename: str):
    with open(path, "rb") as f:
        data = f.read()
    _os.remove(path)
    return StreamingResponse(
        _io.BytesIO(data),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/meta/leagues", tags=["meta"])
def list_leagues():
    try:
        df = db.read_sql(
            "SELECT DISTINCT league FROM matches WHERE league IS NOT NULL ORDER BY league",
            {}
        )
        return {"data": df["league"].tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/meta/seasons", tags=["meta"])
def list_seasons():
    try:
        df = db.read_sql(
            "SELECT DISTINCT season FROM matches WHERE season IS NOT NULL ORDER BY season DESC",
            {}
        )
        return {"data": df["season"].tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports/team/{team_name}/pdf", tags=["reports"])
def team_report_pdf(
    team_name: str,
    season: Optional[str] = Query(None, description="e.g. 25/26"),
    opponent: Optional[str] = Query(None, description="Head-to-head opponent team name"),
    include_players: bool = Query(False, description="Include player detail pages"),
    style_target: Optional[str] = Query(None, description="template:high-press or team:PSG"),
):
    from src.api.report_generator import generate_team_report
    try:
        path = generate_team_report(
            team_name, season=season,
            opponent=opponent,
            include_players=include_players,
            style_target=style_target,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    safe = team_name.lower().replace(" ", "_")
    return _stream_pdf(path, f"{safe}_rapor.pdf")


@app.get("/reports/team/{team_name}/preview", tags=["reports"])
def team_report_preview(
    team_name: str,
    season: Optional[str] = Query(None),
):
    """Return PDF pages as base64 PNGs (in-browser preview)."""
    import tempfile, subprocess, sys
    from src.api.report_generator import generate_team_report

    try:
        pdf_path = generate_team_report(team_name, season=season)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # No pypdf2 + matplotlib for pdf → png conversion; return the PDF directly
    # Preview: thumbnail of the first page
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=96, first_page=1, last_page=1)
        buf = _io.BytesIO()
        images[0].save(buf, format="PNG")
        thumb = _b64.b64encode(buf.getvalue()).decode()
        _os.remove(pdf_path)
        return {"team": team_name, "pages": 7, "thumbnail": thumb}
    except Exception:
        # Without pdf2image only return the download link
        _os.remove(pdf_path)
        return {"team": team_name, "pages": 7,
                "message": "Preview needs the pdf2image library. Use the /pdf endpoint to download the PDF."}


@app.get("/reports/compare/{team_a}/{team_b}/pdf", tags=["reports"])
def comparison_report_pdf(
    team_a: str,
    team_b: str,
    season: Optional[str] = Query(None),
):
    from src.api.report_generator import generate_comparison_report
    try:
        path = generate_comparison_report(team_a, team_b, season=season)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    sa = team_a.lower().replace(" ", "_")
    sb = team_b.lower().replace(" ", "_")
    return _stream_pdf(path, f"{sa}_vs_{sb}_rapor.pdf")


# ---------------------------------------------------------------------------
# Role / Transfer intelligence — imports
# ---------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim


# ---------------------------------------------------------------------------
# Position profiles — tek kaynak src.analysis.profiles
# ---------------------------------------------------------------------------

from src.analysis.profiles import (
    POSITION_PROFILES,
    POS_REQUIRED_DETAILED as _POS_REQUIRED,
    POS_ROLES_DETAILED as _POS_ROLES,
    compute_role_scores as _compute_role_scores,
)
from src.analysis.position_resolver import apply_resolved_position

PROFILE_GROUPS = {
    "Goalkeepers":  ["Goalkeeper"],
    "Center Backs": ["Ball-Playing CB", "Combative CB / Stopper", "Libero / Middle CB", "Wide CB (in 3)"],
    "Midfielders":  ["Defensive Midfielder (#6)", "Attacking Midfielder (#8)", "Deep-Lying Playmaker", "Box-to-Box Midfielder"],
    "Wide Players": ["Full-Back", "Wing-Back", "Classic Winger", "Inverted Winger"],
    "Forwards":     ["Playmaker (#10)", "Target Man (#9)", "Poacher", "Pressing Forward"],
}

def _load_player_pool(
    league: Optional[str],
    season: Optional[str],
    min_minutes: int,
    limit: int = 2000,
) -> pd.DataFrame:
    """Run the same aggregation as player_advanced and apply _add_derived."""
    filters, params = [], {}
    if league:
        filters.append("m.league = :league")
        params["league"] = league
    if season:
        # Accept both '2025-2026' and '25/26'
        parts = str(season).split("-")
        ss = f"{parts[0][2:]}/{parts[1][2:]}" if len(parts) == 2 and len(parts[0]) == 4 else season
        filters.append("m.season IN (:season, :season_ss)")
        params["season"] = season
        params["season_ss"] = ss
    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    params["min_minutes"] = min_minutes
    params["limit"] = limit

    query = f"""
        SELECT
            ps.source_player_id, ps.player_name, ps.team_name, ps.position,
            SUM(ps.minutes_played)                               AS minutes_played,
            COUNT(DISTINCT ps.match_id)                          AS matches_played,
            ROUND(AVG(ps.rating)::numeric, 2)                    AS rating,
            SUM(ps.goals)                                        AS goals,
            SUM(ps.goal_assist)                                  AS goal_assist,
            SUM(ps.expected_goals)                               AS expected_goals,
            SUM(ps.expected_assists)                             AS expected_assists,
            SUM(ps.total_shots)                                  AS total_shots,
            SUM(ps.total_pass)                                   AS total_pass,
            SUM(ps.accurate_pass)                                AS accurate_pass,
            SUM(COALESCE(ps.total_long_balls, 0))                AS total_long_balls,
            SUM(COALESCE(ps.accurate_long_balls, 0))             AS accurate_long_balls,
            SUM(COALESCE(ps.total_cross, 0))                     AS total_cross,
            SUM(COALESCE(ps.accurate_cross, 0))                  AS accurate_cross,
            SUM(COALESCE(ps.key_pass, 0))                        AS key_pass,
            SUM(COALESCE(ps.total_progression, 0))               AS total_progression,
            SUM(COALESCE(ps.progressive_ball_carries_count, 0))  AS progressive_ball_carries_count,
            SUM(COALESCE(ps.duel_won, 0))                        AS duel_won,
            SUM(COALESCE(ps.duel_lost, 0))                       AS duel_lost,
            SUM(COALESCE(ps.aerial_won, 0))                      AS aerial_won,
            SUM(COALESCE(ps.aerial_lost, 0))                     AS aerial_lost,
            SUM(COALESCE(ps.interception_won, 0))                AS interception_won,
            SUM(COALESCE(ps.outfielder_block, 0))                AS outfielder_block,
            SUM(COALESCE(ps.won_contest, 0))                     AS won_contest,
            SUM(COALESCE(ps.challenge_lost, 0))                  AS challenge_lost,
            SUM(COALESCE(ps.was_fouled, 0))                      AS was_fouled,
            SUM(COALESCE(ps.accurate_opposition_half_passes, 0)) AS accurate_opposition_half_passes
        FROM player_stats ps
        JOIN matches m ON m.match_id = ps.match_id
        {where}
        GROUP BY ps.source_player_id, ps.player_name, ps.team_name, ps.position
        HAVING SUM(ps.minutes_played) >= :min_minutes
        ORDER BY AVG(ps.rating) DESC NULLS LAST
        LIMIT :limit
    """
    df = db.read_sql(query, params)
    if not df.empty:
        df = _add_derived(df)
        df = apply_resolved_position(df, id_col="source_player_id",
                                     pos_col="position", season=season)
    return df


# ---------------------------------------------------------------------------
# GET /meta/roles
# ---------------------------------------------------------------------------

@app.get("/meta/roles", tags=["meta"])
def list_roles():
    """Return all position profiles grouped by category."""
    profiles_out = {}
    for role, profile in POSITION_PROFILES.items():
        profiles_out[role] = {
            "group": profile["group"],
            "metrics": {
                key: list(val)
                for key, val in profile["metrics"].items()
            },
        }
    return {"groups": PROFILE_GROUPS, "profiles": profiles_out}


# ---------------------------------------------------------------------------
# GET /stats/role-top10
# ---------------------------------------------------------------------------

@app.get("/stats/role-top10", tags=["stats"])
def role_top10(
    role: str = Query(..., description="Profile name, e.g. 'Ball-Playing CB'"),
    position: Optional[str] = Query(None, description="11-pos: GK/RB/LB/RCB/LCB/CDM/CM/CAM/RW/LW/ST"),
    league: Optional[str] = Query(None),
    season: Optional[str] = Query(None),
    min_minutes: int = Query(450, ge=0),
    limit: int = Query(15, ge=1, le=100),
):
    if role not in POSITION_PROFILES:
        raise HTTPException(status_code=404, detail=f"Role '{role}' not found. See /meta/roles.")

    profile = POSITION_PROFILES[role]
    met_defs = profile["metrics"]

    try:
        df = _load_player_pool(league, season, min_minutes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if df.empty:
        return {"role": role, "metrics": [], "data": []}

    if position:
        df = df[df["position"].str.upper() == position.upper()]
    if df.empty:
        return {"role": role, "metrics": [], "data": []}

    # Compute scores
    scores = _compute_role_scores(df, met_defs)
    df = df.copy()
    df["_score"] = scores

    # Also expose rol_pct and rating_pct
    metric_cols = [col for _, col, _ in met_defs.values() if col in df.columns]
    if metric_cols:
        pct_parts = [df[col].fillna(0).rank(pct=True) * 100 for col in metric_cols]
        df["_rol_pct"] = pd.concat(pct_parts, axis=1).mean(axis=1).round(2)
    else:
        df["_rol_pct"] = 0.0

    if "rating" in df.columns:
        df["_rating_pct"] = (df["rating"].fillna(0).rank(pct=True) * 100).round(2)
    else:
        df["_rating_pct"] = 0.0

    df = df.sort_values("_score", ascending=False).head(limit)

    metrics_meta = [
        {"key": key, "label": label, "col": col, "kind": kind}
        for key, (label, col, kind) in met_defs.items()
    ]

    return {
        "role": role,
        "metrics": metrics_meta,
        "data": _df_to_list(df),
    }


# ---------------------------------------------------------------------------
# GET /transfer/similar
# ---------------------------------------------------------------------------

@app.get("/transfer/similar", tags=["transfer"])
def transfer_similar(
    player_id: int = Query(..., description="source_player_id of the target player"),
    role: str = Query(..., description="Profile name, e.g. 'Ball-Playing CB'"),
    league: Optional[str] = Query(None),
    season: Optional[str] = Query(None),
    min_minutes: int = Query(450, ge=0),
    n: int = Query(10, ge=1, le=50),
):
    if role not in POSITION_PROFILES:
        raise HTTPException(status_code=404, detail=f"Role '{role}' not found. See /meta/roles.")

    met_defs = POSITION_PROFILES[role]["metrics"]

    try:
        df = _load_player_pool(league, season, min_minutes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if df.empty:
        raise HTTPException(status_code=404, detail="No players found with given filters.")

    target_rows = df[df["source_player_id"] == player_id]
    if target_rows.empty:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found in pool.")

    # Feature matrix — only available metric cols
    feat_cols = [col for _, col, _ in met_defs.values() if col in df.columns]
    if not feat_cols:
        raise HTTPException(status_code=400, detail="No feature columns available for this role.")

    df = df.reset_index(drop=True)
    target_idx = df[df["source_player_id"] == player_id].index[0]

    X = df[feat_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    target_vec = X_scaled[target_idx].reshape(1, -1)
    sims = _cosine_sim(target_vec, X_scaled)[0]  # shape (n_players,)

    df = df.copy()
    df["similarity"] = (sims * 100).round(2)

    # Remove target from results
    others = df[df["source_player_id"] != player_id].copy()
    others = others.sort_values("similarity", ascending=False).head(n)

    target_info = _df_to_list(df[df["source_player_id"] == player_id])[0]

    return {
        "target": target_info,
        "similar": _df_to_list(others),
    }


# ---------------------------------------------------------------------------
# GET /transfer/star-picks
# ---------------------------------------------------------------------------

@app.get("/transfer/star-picks", tags=["transfer"])
def transfer_star_picks(
    role: str = Query(..., description="Profile name, e.g. 'Ball-Playing CB'"),
    league: Optional[str] = Query(None),
    season: Optional[str] = Query(None),
    min_minutes: int = Query(450, ge=0),
    limit: int = Query(15, ge=1, le=100),
):
    if role not in POSITION_PROFILES:
        raise HTTPException(status_code=404, detail=f"Role '{role}' not found. See /meta/roles.")

    met_defs = POSITION_PROFILES[role]["metrics"]

    try:
        df = _load_player_pool(league, season, min_minutes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if df.empty:
        return {"role": role, "data": []}

    df = df.copy()
    df["_score"] = _compute_role_scores(df, met_defs)

    # Team average rating
    if "rating" in df.columns and "team_name" in df.columns:
        team_avg = df.groupby("team_name")["rating"].mean().rename("_team_avg_rating")
        df = df.join(team_avg, on="team_name")
        # Rank teams: lower rank_pct = weaker team
        team_ranks = df.groupby("team_name")["_team_avg_rating"].first().rank(pct=True)
        df["_team_rank_pct"] = df["team_name"].map(team_ranks)
        df["transferable"] = df["_team_rank_pct"] <= 0.45
        df["team_tier"] = pd.cut(
            df["_team_rank_pct"],
            bins=[0, 0.33, 0.66, 1.0],
            labels=["Bottom", "Mid", "Top"],
            include_lowest=True,
        ).astype(str)
    else:
        df["transferable"] = False
        df["team_tier"] = "Unknown"

    df = df.sort_values("_score", ascending=False).head(limit)

    return {"role": role, "data": _df_to_list(df)}


# ---------------------------------------------------------------------------
# GET /transfer/budget-gems
# ---------------------------------------------------------------------------

@app.get("/transfer/budget-gems", tags=["transfer"])
def transfer_budget_gems(
    role: str = Query(..., description="Profile name, e.g. 'Ball-Playing CB'"),
    league: Optional[str] = Query(None),
    season: Optional[str] = Query(None),
    min_minutes: int = Query(450, ge=0),
    limit: int = Query(10, ge=1, le=100),
    reference_team: Optional[str] = Query(None, description="Reference team — show players from clubs ranked below this team"),
):
    if role not in POSITION_PROFILES:
        raise HTTPException(status_code=404, detail=f"Role '{role}' not found. See /meta/roles.")

    met_defs = POSITION_PROFILES[role]["metrics"]

    try:
        df = _load_player_pool(league, season, min_minutes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if df.empty:
        return {"role": role, "data": [], "budget_threshold": None}

    df = df.copy()
    df["_score"] = _compute_role_scores(df, met_defs)

    # Compute team rank percentile (higher = better team)
    budget_threshold = 0.5
    ref_team_pct = None
    if "rating" in df.columns and "team_name" in df.columns:
        team_avg = df.groupby("team_name")["rating"].mean().rename("_team_avg_rating")
        df = df.join(team_avg, on="team_name")
        team_ranks = df.groupby("team_name")["_team_avg_rating"].first().rank(pct=True)
        df["_team_rank_pct"] = df["team_name"].map(team_ranks)

        if reference_team:
            # Find reference team's percentile rank
            ref_lower = reference_team.lower()
            match = df[df["team_name"].str.lower() == ref_lower]
            if not match.empty:
                ref_team_pct = float(match["_team_rank_pct"].iloc[0])
                # Show players from teams strictly below the reference team
                budget_threshold = ref_team_pct
    else:
        df["_team_rank_pct"] = 0.0

    # Budget gems: players from clubs ranked below the threshold
    df = df[df["_team_rank_pct"] < budget_threshold]
    df = df.sort_values("_score", ascending=False).head(limit)

    out = _df_to_list(df)
    return {
        "role": role,
        "data": out,
        "budget_threshold": round(budget_threshold, 2),
        "reference_team": reference_team,
        "reference_team_pct": round(ref_team_pct, 2) if ref_team_pct is not None else None,
    }


# ---------------------------------------------------------------------------
# GET /transfer/squad-needs/{team_name}
# ---------------------------------------------------------------------------

@app.get("/transfer/squad-needs/{team_name}", tags=["transfer"])
def transfer_squad_needs(
    team_name: str,
    season: Optional[str] = Query(None),
    min_minutes: int = Query(450, ge=0),
):
    try:
        df = _load_player_pool(league=None, season=season, min_minutes=min_minutes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if df.empty:
        raise HTTPException(status_code=404, detail="No player data found.")

    # Filter to target team
    team_df = df[df["team_name"].str.lower() == team_name.lower()]
    if team_df.empty:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found in pool.")

    # Count players per detailed (11) position
    pos_counts: dict[str, int] = {}
    for pos in _POS_REQUIRED.keys():
        pos_counts[pos] = int((team_df["position"].str.upper() == pos).sum())

    # Identify gaps
    gaps = []
    for pos, required in _POS_REQUIRED.items():
        current = pos_counts.get(pos, 0)
        gap = max(0, required - current)
        status = "ok" if gap == 0 else "need"
        gaps.append({
            "pos": pos,
            "current": current,
            "required": required,
            "gap": gap,
            "status": status,
        })

    # Build suggestions for each gap
    other_df = df[df["team_name"].str.lower() != team_name.lower()]
    suggestions: dict[str, dict] = {}

    for gap_info in gaps:
        if gap_info["status"] == "ok":
            continue
        pos = gap_info["pos"]
        roles_for_pos = _POS_ROLES.get(pos, [])
        if not roles_for_pos:
            suggestions[pos] = {}
            continue

        pos_others = other_df[other_df["position"].str.upper() == pos]
        role_suggestions: dict[str, list] = {}

        for role in roles_for_pos:
            if role not in POSITION_PROFILES:
                continue
            met_defs = POSITION_PROFILES[role]["metrics"]
            if pos_others.empty:
                role_suggestions[role] = []
                continue
            role_df = pos_others.copy().reset_index(drop=True)
            role_df["_score"] = _compute_role_scores(role_df, met_defs)
            top5 = role_df.sort_values("_score", ascending=False).head(5)
            role_suggestions[role] = _df_to_list(top5)

        suggestions[pos] = role_suggestions

    return {
        "team": team_name,
        "gaps": gaps,
        "suggestions": suggestions,
    }


# ---------------------------------------------------------------------------
# Style — templates, fingerprint, cluster scatter
# ---------------------------------------------------------------------------

from src.analysis.style import (
    cluster_teams as _cluster_teams,
    style_fingerprint as _style_fingerprint,
)
from src.analysis.style_templates import (
    list_templates as _list_style_templates,
    parse_style_target as _parse_style_target,
)
from src.analysis.insights import (
    team_strengths_weaknesses as _team_sw,
    h2h_tactical_recommendations as _h2h_reco,
)
from src.analysis.transfer import (
    get_formation_slots as _get_formation_slots,
    evaluate_squad_fit as _evaluate_squad_fit,
)
from fastapi import Body


@app.get("/meta/style-templates", tags=["meta"])
def meta_style_templates():
    return {"templates": _list_style_templates()}


@app.get("/meta/formations", tags=["meta"])
def meta_formations():
    forms = ["4-3-3", "4-2-3-1", "3-5-2", "4-4-2"]
    return {"formations": [{"id": f, "slots": _get_formation_slots(f)} for f in forms]}


@app.get("/stats/style-scatter", tags=["stats"])
def stats_style_scatter(
    league: Optional[str] = None,
    season: Optional[str] = None,
):
    try:
        ts = db.get_team_stats(league, season)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if ts.empty:
        return {"count": 0, "data": []}
    try:
        clustered = _cluster_teams(ts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"clustering failed: {e}")
    out_cols = ["team_name", "league", "season", "PC1", "PC2", "cluster"]
    out = clustered[[c for c in out_cols if c in clustered.columns]]
    return {"count": len(out), "data": _df_to_list(out)}


@app.get("/teams/{team_name}/style-fingerprint", tags=["teams"])
def team_style_fingerprint(
    team_name: str,
    league: Optional[str] = None,
    season: Optional[str] = None,
):
    try:
        ts = db.get_team_stats(league, season)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if ts.empty:
        raise HTTPException(status_code=404, detail="No team stats.")
    fp = _style_fingerprint(ts, team_name)
    if fp.empty:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found.")
    return {"team": team_name, "fingerprint": {k: float(v) for k, v in fp.items()}}


@app.get("/teams/{team_name}/why-successful", tags=["teams"])
def team_why_successful(
    team_name: str,
    league: Optional[str] = None,
    season: Optional[str] = None,
):
    try:
        ts = db.get_team_stats(league, season)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if ts.empty:
        raise HTTPException(status_code=404, detail="No team stats available.")

    fp = _style_fingerprint(ts, team_name)
    if fp.empty:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found.")

    sw = _team_sw(fp)

    # Standing (from the league table)
    row = ts[ts["team_name"] == team_name].iloc[0].to_dict() if not ts[ts["team_name"] == team_name].empty else {}
    summary_cards = {
        "matches_played": row.get("matches_played"),
        "wins": row.get("wins"),
        "draws": row.get("draws"),
        "losses": row.get("losses"),
        "points": row.get("points"),
        "goals_for": row.get("goals_for"),
        "goals_against": row.get("goals_against"),
        "xg": row.get("xg"),
        "xg_against": row.get("xg_against"),
        "xg_diff": row.get("xg_diff"),
    }
    summary_cards = {k: _to_python(v) for k, v in summary_cards.items()}

    return {
        "team": team_name,
        "summary": summary_cards,
        "fingerprint": {k: float(v) for k, v in fp.items()},
        "strengths": sw["strengths"],
        "weaknesses": sw["weaknesses"],
        "summary_text": sw["summary"],
    }


@app.get("/h2h/{team_a}/{team_b}", tags=["teams"])
def h2h_analysis(
    team_a: str,
    team_b: str,
    league: Optional[str] = None,
    season: Optional[str] = None,
):
    try:
        ts = db.get_team_stats(league, season)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if ts.empty:
        raise HTTPException(status_code=404, detail="No team stats.")

    fp_a = _style_fingerprint(ts, team_a)
    fp_b = _style_fingerprint(ts, team_b)
    if fp_a.empty or fp_b.empty:
        raise HTTPException(status_code=404, detail="One or both teams not found.")

    reco = _h2h_reco(fp_a, fp_b, our_name=team_a, their_name=team_b)
    reco_inv = _h2h_reco(fp_b, fp_a, our_name=team_b, their_name=team_a)

    # Past scores
    hist_q = """
        SELECT match_id, season, round, match_date::text AS date,
               home_team_name, away_team_name, home_goals, away_goals
        FROM matches
        WHERE (home_team_name = :a AND away_team_name = :b)
           OR (home_team_name = :b AND away_team_name = :a)
        ORDER BY match_date DESC NULLS LAST
        LIMIT 20
    """
    try:
        hist = db.read_sql(hist_q, {"a": team_a, "b": team_b})
        history = _df_to_list(hist) if not hist.empty else []
    except Exception:
        history = []

    return {
        "team_a": team_a,
        "team_b": team_b,
        "fingerprint_a": {k: float(v) for k, v in fp_a.items()},
        "fingerprint_b": {k: float(v) for k, v in fp_b.items()},
        "recommendations_for_a": reco,
        "recommendations_for_b": reco_inv,
        "history": history,
    }


@app.post("/transfer/squad-fit", tags=["transfer"])
def transfer_squad_fit(payload: dict = Body(...)):
    """Body: {team, formation, style_target (template:id|team:name|null),
             slot_assignments: {slot_id: player_id?}, league?, season?, min_minutes?}.

    Response: per slot {assigned_player, fit_pct, reason}.
    """
    team = payload.get("team")
    formation = payload.get("formation", "4-3-3")
    style_target = payload.get("style_target")
    slot_assignments = payload.get("slot_assignments") or {}
    league = payload.get("league")
    season = payload.get("season")
    min_minutes = int(payload.get("min_minutes", 0))

    if not team:
        raise HTTPException(status_code=400, detail="team required")

    try:
        pool = _load_player_pool(league=league, season=season, min_minutes=min_minutes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    team_df = pool[pool["team_name"].str.lower() == team.lower()] if not pool.empty else pool

    # Target style vector
    target_vector = None
    try:
        ts = db.get_team_stats(league, season)
        target_vector = _parse_style_target(style_target, ts) if style_target else None
    except Exception:
        target_vector = None

    slots = _evaluate_squad_fit(team_df, formation, target_vector, slot_assignments)

    # Suggestions for empty slots: from players outside the team, top 5 by role score for slot.suggested_role.
    other_df = pool[pool["team_name"].str.lower() != team.lower()] if not pool.empty else pool
    for slot in slots:
        if slot["assigned_player"] is None and not other_df.empty:
            role = slot["suggested_role"]
            if role in POSITION_PROFILES:
                met_defs = POSITION_PROFILES[role]["metrics"]
                pos_pool = other_df[other_df["position"].str.upper() == slot["position_code"]].copy()
                if not pos_pool.empty:
                    pos_pool = pos_pool.reset_index(drop=True)
                    pos_pool["_score"] = _compute_role_scores(pos_pool, met_defs)
                    top5 = pos_pool.nlargest(5, "_score")[
                        ["source_player_id", "player_name", "team_name", "position",
                         "matches_played", "rating", "_score"]
                    ]
                    slot["suggestions"] = _df_to_list(top5)
                else:
                    slot["suggestions"] = []
            else:
                slot["suggestions"] = []
        else:
            slot["suggestions"] = []

    return {
        "team": team,
        "formation": formation,
        "style_target": style_target,
        "slots": slots,
    }
