"""Edge agregatları — maç-bazlı lineup verisinden ikili istatistikler.

Her `(player_a, player_b, season)` çifti için:
- minutes_together : iki oyuncunun aynı takımda aynı anda sahada olduğu dakika
- matches_together
- joint_g, joint_a, joint_xg, joint_xa (event verisi varsa)
- per90_joint_output = (joint_g + joint_a) / minutes_together * 90

`teammate` ve `opponent` kenarları ayrı çıkar. Min. filtre: ≥ 90 dakika birlikte.

Çıktı: data/graph/edges_<season>.parquet (teammate ve opponent ayrı dosyalar).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

from src import db

GRAPH_DIR = Path("data/graph")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

MIN_MINUTES_TOGETHER = 90


def _season_where(season: str, league: str | None = None) -> tuple[str, dict]:
    """matches.season hem '25/26' hem '2025-2026' formatını kabul eder."""
    where = []
    params: dict = {}
    if season:
        parts = str(season).split("-")
        ss = f"{parts[0][2:]}/{parts[1][2:]}" if len(parts) == 2 and len(parts[0]) == 4 else season
        where.append("(m.season = :season_full OR m.season = :season_ss)")
        params["season_full"] = season
        params["season_ss"] = ss
    if league:
        where.append("m.league = :league")
        params["league"] = league
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    return where_sql, params


def _load_lineups(season: str, league: str | None = None) -> pd.DataFrame:
    """player_stats'ten lineup-benzeri DataFrame inşa eder.

    minutes_on/off türetilir:
      - starter (substitute=False): minutes_on=0, minutes_off=minutes_played
      - sub (substitute=True): minutes_on=90-minutes_played, minutes_off=90
    NULL minutes_played olan kayıtlar (sahaya çıkmamış yedekler) elenir.
    """
    where_sql, params = _season_where(season, league)
    sql = f"""
        SELECT
            p.match_id,
            m.league,
            CASE WHEN m.season LIKE '__/__' THEN
                '20' || SPLIT_PART(m.season, '/', 1) || '-20' || SPLIT_PART(m.season, '/', 2)
                ELSE m.season END                              AS season,
            m.match_date,
            'ss_t_' || p.team_id::text                          AS team_id,
            'ss_' || p.source_player_id::text                AS player_id,
            p.player_name,
            CASE WHEN p.substitute
                 THEN GREATEST(0, 90 - p.minutes_played)
                 ELSE 0 END                                     AS minutes_on,
            CASE WHEN p.substitute
                 THEN 90
                 ELSE LEAST(p.minutes_played, 90) END           AS minutes_off,
            p.position,
            NOT COALESCE(p.substitute, false)                   AS is_starter,
            'ss_t_' || m.home_team_id::text                     AS home_team_id,
            'ss_t_' || m.away_team_id::text                     AS away_team_id
        FROM player_stats p
        JOIN matches m ON m.match_id = p.match_id
        {where_sql}
        AND p.minutes_played IS NOT NULL AND p.minutes_played > 0
    """ if where_sql else f"""
        SELECT
            p.match_id,
            m.league,
            CASE WHEN m.season LIKE '__/__' THEN
                '20' || SPLIT_PART(m.season, '/', 1) || '-20' || SPLIT_PART(m.season, '/', 2)
                ELSE m.season END                              AS season,
            m.match_date,
            'ss_t_' || p.team_id::text                          AS team_id,
            'ss_' || p.source_player_id::text                AS player_id,
            p.player_name,
            CASE WHEN p.substitute
                 THEN GREATEST(0, 90 - p.minutes_played)
                 ELSE 0 END                                     AS minutes_on,
            CASE WHEN p.substitute
                 THEN 90
                 ELSE LEAST(p.minutes_played, 90) END           AS minutes_off,
            p.position,
            NOT COALESCE(p.substitute, false)                   AS is_starter,
            'ss_t_' || m.home_team_id::text                     AS home_team_id,
            'ss_t_' || m.away_team_id::text                     AS away_team_id
        FROM player_stats p
        JOIN matches m ON m.match_id = p.match_id
        WHERE p.minutes_played IS NOT NULL AND p.minutes_played > 0
    """
    return db.read_sql(sql, params)


def _load_events(season: str, league: str | None = None) -> pd.DataFrame:
    """player_stats'ten event DF — joint output hesaplaması için."""
    where_sql, params = _season_where(season, league)
    sql = f"""
        SELECT
            p.match_id,
            'ss_' || p.source_player_id::text   AS player_id,
            COALESCE(p.goals, 0)                    AS goals,
            COALESCE(p.goal_assist, 0)              AS assists,
            COALESCE(p.expected_goals, 0)           AS xg,
            COALESCE(p.expected_assists, 0)         AS xa
        FROM player_stats p
        JOIN matches m ON m.match_id = p.match_id
        {where_sql}
    """
    try:
        return db.read_sql(sql, params)
    except Exception:
        return pd.DataFrame(columns=["match_id", "player_id", "goals", "assists", "xg", "xa"])


def _overlap(a_on: int, a_off: int, b_on: int, b_off: int) -> int:
    return max(0, min(a_off, b_off) - max(a_on, b_on))


def _per_match_pairs(group: pd.DataFrame, kind: str = "teammate") -> pd.DataFrame:
    """Tek bir maç için pairwise overlap üretir.

    kind='teammate': aynı team_id içindeki çiftler
    kind='opponent': farklı team_id'deki çiftler (home × away)
    """
    rows = []
    if kind == "teammate":
        for _, team_df in group.groupby("team_id"):
            players = team_df[["player_id", "minutes_on", "minutes_off"]].values
            for (pa, on_a, off_a), (pb, on_b, off_b) in combinations(players, 2):
                if pa == pb:
                    continue
                ov = _overlap(on_a, off_a, on_b, off_b)
                if ov <= 0:
                    continue
                pid_a, pid_b = sorted([pa, pb])
                rows.append((pid_a, pid_b, ov, 1))
    else:
        teams = list(group.groupby("team_id"))
        if len(teams) != 2:
            return pd.DataFrame(columns=["player_a", "player_b", "minutes_together", "matches_together"])
        (_, ta), (_, tb) = teams
        for pa, on_a, off_a in ta[["player_id", "minutes_on", "minutes_off"]].values:
            for pb, on_b, off_b in tb[["player_id", "minutes_on", "minutes_off"]].values:
                ov = _overlap(on_a, off_a, on_b, off_b)
                if ov <= 0:
                    continue
                pid_a, pid_b = sorted([pa, pb])
                rows.append((pid_a, pid_b, ov, 1))

    return pd.DataFrame(rows, columns=["player_a", "player_b", "minutes_together", "matches_together"])


def _solo_baseline(lineups: pd.DataFrame, events: pd.DataFrame) -> dict[str, float]:
    """Her oyuncu için solo per90 G+A baseline (tüm sezon, tüm maçlar).

    Uplift hedef değişkeni için kullanılır:
        per90_uplift = observed_pair_per90 − (solo_a + solo_b)

    Pozitif uplift = "birlikteyken beklenenden çok üretiyorlar" (sinerji).
    Negatif uplift = "birbirlerini yokediyorlar".

    Not: Pair maçları da baseline'a dahil — pragmatik kabulleniş. Oyuncu
    başına maç sayısı yüksek olduğu için bias küçük; leave-one-out hesap
    çok pahalı olur.
    """
    if events.empty:
        return {}
    lp = lineups[["match_id", "player_id", "minutes_on", "minutes_off"]].copy()
    lp["mins"] = (lp["minutes_off"] - lp["minutes_on"]).clip(lower=0)
    merged = lp.merge(events[["match_id", "player_id", "goals", "assists"]],
                      on=["match_id", "player_id"], how="left")
    merged["ga"] = merged["goals"].fillna(0) + merged["assists"].fillna(0)
    agg = merged.groupby("player_id", as_index=False).agg(
        ga=("ga", "sum"), mins=("mins", "sum")
    )
    agg["per90"] = np.where(agg["mins"] > 0, agg["ga"] / agg["mins"] * 90.0, 0.0)
    return dict(zip(agg["player_id"], agg["per90"]))


def build_edges(season: str, league: str | None = None) -> dict[str, pd.DataFrame]:
    """Sezon için teammate + opponent edge agregatlarını döndürür."""
    lineups = _load_lineups(season, league)
    if lineups.empty:
        print(f"[edges] {season}: lineup verisi yok")
        return {"teammate": pd.DataFrame(), "opponent": pd.DataFrame()}

    events = _load_events(season, league)
    has_events = not events.empty
    solo = _solo_baseline(lineups, events) if has_events else {}

    teammate_rows: list[pd.DataFrame] = []
    opponent_rows: list[pd.DataFrame] = []

    for match_id, g in lineups.groupby("match_id"):
        teammate_rows.append(_per_match_pairs(g, kind="teammate").assign(match_id=match_id))
        opponent_rows.append(_per_match_pairs(g, kind="opponent").assign(match_id=match_id))

    teammate = pd.concat(teammate_rows, ignore_index=True) if teammate_rows else pd.DataFrame()
    opponent = pd.concat(opponent_rows, ignore_index=True) if opponent_rows else pd.DataFrame()

    teammate = _aggregate(teammate, events if has_events else None, solo)
    opponent = _aggregate(opponent, None, {})

    # Min filtre
    if not teammate.empty:
        teammate = teammate[teammate["minutes_together"] >= MIN_MINUTES_TOGETHER].copy()
    if not opponent.empty:
        opponent = opponent[opponent["minutes_together"] >= MIN_MINUTES_TOGETHER].copy()

    teammate["season"] = season
    opponent["season"] = season

    out_t = GRAPH_DIR / f"edges_teammate_{season}.parquet"
    out_o = GRAPH_DIR / f"edges_opponent_{season}.parquet"
    teammate.to_parquet(out_t, index=False)
    opponent.to_parquet(out_o, index=False)
    print(f"[edges] teammate={len(teammate)} → {out_t}")
    print(f"[edges] opponent={len(opponent)} → {out_o}")

    return {"teammate": teammate, "opponent": opponent}


def _aggregate(pair_df: pd.DataFrame, events: pd.DataFrame | None,
               solo: dict[str, float] | None = None) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame(columns=[
            "player_a", "player_b", "minutes_together", "matches_together",
            "joint_g", "joint_a", "joint_xg", "joint_xa",
            "per90_joint_output", "expected_per90", "per90_uplift",
        ])

    agg = pair_df.groupby(["player_a", "player_b"], as_index=False).agg(
        minutes_together=("minutes_together", "sum"),
        matches_together=("matches_together", "sum"),
    )

    # Joint output — event verisi varsa, her maç için iki oyuncuya da ait
    # (goals + assists)'i topla; iki oyuncunun toplamını ortak çıktı say.
    if events is not None and not events.empty and "match_id" in pair_df.columns:
        ev = events.copy()
        ev["ga"] = ev["goals"].fillna(0) + ev["assists"].fillna(0)
        ev["xga"] = ev["xg"].fillna(0) + ev["xa"].fillna(0)

        pa_ev = pair_df.merge(ev, left_on=["match_id", "player_a"], right_on=["match_id", "player_id"], how="left")
        pa_ev = pa_ev.rename(columns={"goals": "g_a", "assists": "a_a", "xg": "xg_a", "xa": "xa_a"})
        full = pa_ev.merge(ev, left_on=["match_id", "player_b"], right_on=["match_id", "player_id"], how="left",
                           suffixes=("", "_b"))
        full = full.rename(columns={"goals": "g_b", "assists": "a_b", "xg": "xg_b", "xa": "xa_b"})

        joint = full.groupby(["player_a", "player_b"], as_index=False).agg(
            joint_g=("g_a", "sum"),
            joint_g_b=("g_b", "sum"),
            joint_a=("a_a", "sum"),
            joint_a_b=("a_b", "sum"),
            joint_xg=("xg_a", "sum"),
            joint_xg_b=("xg_b", "sum"),
            joint_xa=("xa_a", "sum"),
            joint_xa_b=("xa_b", "sum"),
        )
        joint["joint_g"] = joint["joint_g"].fillna(0) + joint["joint_g_b"].fillna(0)
        joint["joint_a"] = joint["joint_a"].fillna(0) + joint["joint_a_b"].fillna(0)
        joint["joint_xg"] = joint["joint_xg"].fillna(0) + joint["joint_xg_b"].fillna(0)
        joint["joint_xa"] = joint["joint_xa"].fillna(0) + joint["joint_xa_b"].fillna(0)
        joint = joint[["player_a", "player_b", "joint_g", "joint_a", "joint_xg", "joint_xa"]]

        agg = agg.merge(joint, on=["player_a", "player_b"], how="left")
    else:
        agg["joint_g"] = 0.0
        agg["joint_a"] = 0.0
        agg["joint_xg"] = 0.0
        agg["joint_xa"] = 0.0

    for c in ("joint_g", "joint_a", "joint_xg", "joint_xa"):
        agg[c] = agg[c].fillna(0.0)

    agg["per90_joint_output"] = (
        (agg["joint_g"] + agg["joint_a"]) / agg["minutes_together"] * 90.0
    ).where(agg["minutes_together"] > 0, 0.0)

    # Uplift = observed_pair_per90 − (solo_a + solo_b). Gerçek sinerji sinyali.
    if solo:
        agg["expected_per90"] = (
            agg["player_a"].map(solo).fillna(0.0)
            + agg["player_b"].map(solo).fillna(0.0)
        )
        agg["per90_uplift"] = agg["per90_joint_output"] - agg["expected_per90"]
    else:
        agg["expected_per90"] = 0.0
        agg["per90_uplift"] = 0.0

    return agg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="2024-2025")
    parser.add_argument("--league", default=None)
    args = parser.parse_args()
    build_edges(args.season, args.league)
