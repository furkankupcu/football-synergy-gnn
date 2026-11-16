"""Tactical Lab analysis endpoint.

POST /tactical/analyze — anchor team + lineup + style target → full analysis package:
  - team_style_fit (overall %)
  - per_player_fit (fit % per slot + metric attribution)
  - pairwise_synergy (GNN predict_edge between the placed players)
  - replacements (top-K transfer suggestions per slot, sorted by combined score)
"""

from __future__ import annotations

from typing import Optional
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, Body, HTTPException
from sqlalchemy import text

from src import db
from src.api.gnn_routes import _engine, _resolve_season
from src.analysis.style_templates import parse_style_target, get_template_vector
from src.analysis.style import style_fingerprint
from src.analysis.profiles import compute_role_scores
from src.tactical.style_fit import (
    POS_METRICS, aggregate_player_metrics, style_fit_for_lineup,
)
from src.tactical.recommend import POS_FAMILY, position_match, gnn_synergy_avg

router = APIRouter(prefix="/tactical", tags=["tactical"])


# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _player_pool(season: str) -> pd.DataFrame:
    """Load all players of the season with per-90 metrics (cached).

    Resolves the existing columns via information_schema first; missing metrics are left out of the SELECT.
    """
    all_metrics = sorted({m for ms in POS_METRICS.values() for (m, _) in ms})
    try:
        cols_df = db.read_sql(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='player_stats'"
        )
        existing = set(cols_df["column_name"].tolist())
    except Exception:
        existing = set()
    base = ["source_player_id", "player_name", "team_name", "position",
            "minutes_played", "rating"]
    select_cols = [c for c in base if c in existing] + [m for m in all_metrics if m in existing]
    if not select_cols:
        return pd.DataFrame()
    sql = f"SELECT {', '.join(select_cols)} FROM player_stats WHERE minutes_played IS NOT NULL"
    try:
        return db.read_sql(sql)
    except Exception as e:
        print(f"[tactical] _player_pool fallback: {e}")
        return pd.DataFrame()


def _team_stats_for_season(season: str) -> pd.DataFrame:
    """Team stats for the style team vector (match aggregate)."""
    try:
        return db.read_sql(f"""
            SELECT * FROM team_stats
            WHERE season = '{season}'
        """)
    except Exception:
        return pd.DataFrame()


def _style_vector(style: dict | None, season: str) -> Optional[pd.Series]:
    """Parse the style target: {mode: 'template'|'team', target: id|team_name} → percentile vector."""
    if not style:
        return None
    mode, target = style.get("mode"), style.get("target")
    if not mode or not target:
        return None
    if mode == "template":
        try:
            return get_template_vector(target)
        except ValueError:
            return None
    if mode == "team":
        team_stats = _team_stats_for_season(season)
        if team_stats.empty:
            return None
        try:
            return style_fingerprint(team_stats, target)
        except Exception:
            return None
    return None


def _pairwise_synergy(eng, placed: list[tuple[str, str]]) -> list[dict]:
    """placed = [(slot, pid)]; predict_edge for every pair."""
    n = len(placed)
    if n < 2:
        return []
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            (sa, pa), (sb, pb) = placed[i], placed[j]
            ai = eng.pid_to_idx.get(pa)
            bi = eng.pid_to_idx.get(pb)
            if ai is None or bi is None:
                continue
            with torch.no_grad():
                s = eng.model.predict_edge(
                    eng._emb,
                    torch.tensor([ai], dtype=torch.long, device=eng.device),
                    torch.tensor([bi], dtype=torch.long, device=eng.device),
                ).item()
            out.append({
                "a": pa, "b": pb, "slot_a": sa, "slot_b": sb,
                "label": f"{sa}↔{sb}", "score": round(float(s), 4),
            })
    return out


# ------------------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------------------

@router.post("/analyze")
def analyze(payload: dict = Body(...)):
    """Full style-driven squad analysis.

    Body:
      {
        "anchor_team": "Fenerbahçe",
        "lineup":     {"GK": "ss_39573", "RB": "ss_252821", ...},
        "style":      {"mode": "template"|"team", "target": "high-press"|"Manchester City"} | null,
        "season":     "2025-2026",
        "filters":    {"min_minutes": 900, "max_age": 99},
        "top_k":      5
      }
    """
    season = _resolve_season(payload.get("season"))
    eng = _engine(season)

    lineup_raw: dict = payload.get("lineup") or {}
    # Normalise: 'ss_xxx' → 'ss_xxx' (string); ints get the prefix as well
    lineup: dict[str, str] = {}
    for slot, pid in lineup_raw.items():
        if not pid:
            continue
        spid = str(pid)
        if not spid.startswith("ss_"):
            spid = f"ss_{spid}"
        lineup[slot.upper()] = spid

    filters = payload.get("filters") or {}
    min_minutes = int(filters.get("min_minutes", 900))
    top_k = int(payload.get("top_k", 5))

    # 1) Style vector
    style_vec = _style_vector(payload.get("style"), season)

    # 2) Oyuncu havuzu (per-90)
    pool_raw = _player_pool(season)
    # Aggregate match-level rows to per-90 per player
    all_metrics = sorted({m for ms in POS_METRICS.values() for (m, _) in ms})
    metrics_present = [c for c in all_metrics if c in pool_raw.columns]
    per90 = aggregate_player_metrics(pool_raw, metrics_present) if metrics_present else pd.DataFrame()
    # Same pool for league percentiles (simple; more accurate: split by position)
    league_pool = per90

    # 3) Per-player fit
    style_pkg = style_fit_for_lineup(
        lineup=lineup,
        player_metrics=per90,
        style_vector=style_vec,
        league_pool=league_pool,
    )

    # 4) Pairwise synergy
    placed_pairs = [(s, p) for s, p in lineup.items() if p in eng.pid_to_idx]
    pair_syn = _pairwise_synergy(eng, placed_pairs)

    # 5) Replacements: top-K per slot
    # Join meta onto the pool: player name + team + 11-pos (from the engine)
    if not per90.empty:
        meta = pool_raw.groupby("source_player_id").agg(
            player_name=("player_name", "last"),
            team_name=("team_name", "last"),
        )
        per90 = per90.join(meta, how="left")
        # 11-pos: the GNN engine's position map (instead of G/D/M/F)
        eng_pos = {int(pid.replace("ss_", "")): pos
                   for pid, pos in zip(eng.player_ids, eng.positions)
                   if pid.startswith("ss_") and pid[3:].isdigit()}
        per90["position"] = per90.index.map(lambda i: eng_pos.get(int(i)))

    # role_score = the user's compute_role_scores; it uses the main per-match df.
    # Here the per-90 frame has one row per player; pragmatic: average rating as a proxy.
    try:
        rating_avg = pool_raw.groupby("source_player_id")["rating"].mean()
        per90 = per90.join(rating_avg.rename("rating"), how="left")
    except Exception:
        per90["rating"] = np.nan

    # style_fit: per player, a mini score with the POS_METRICS that best fit their position
    style_fit_col = pd.Series(index=per90.index, dtype=float)
    if not per90.empty:
        for spid, row in per90.iterrows():
            pos = (row.get("position") or "").upper()
            metrics = POS_METRICS.get(pos, POS_METRICS.get("CM"))
            scores = []
            for col, _dir in metrics:
                if col not in per90.columns:
                    continue
                val = row[col]
                if pd.isna(val):
                    continue
                pool_vals = league_pool[col].dropna()
                if len(pool_vals) < 5:
                    continue
                pct = float((pool_vals < val).mean()) * 100
                scores.append(pct)
            if scores:
                style_fit_col.at[spid] = float(np.mean(scores))
    per90["style_fit"] = style_fit_col

    # role_score: rating + position-typical weighted percentile of per-90 metrics
    # Pragmatic: same base as style_fit + rating normalisation
    if "rating" in per90.columns:
        rt = per90["rating"].rank(pct=True) * 100
        per90["role_score"] = (per90["style_fit"].fillna(50) * 0.7 + rt.fillna(50) * 0.3).round(1)
    else:
        per90["role_score"] = per90["style_fit"]

    # Filtre: min_minutes
    if "minutes" in per90.columns:
        eligible = per90[per90["minutes"] >= min_minutes].copy()
    else:
        eligible = per90.copy()

    placed_pids = set(lineup.values())
    replacements: dict[str, list[dict]] = {}
    for slot, pid in lineup.items():
        # Candidate pool: same position family (flexible)
        allowed = POS_FAMILY.get(slot, {slot})
        cand = eligible[eligible["position"].fillna("").str.upper().isin(allowed)].copy()
        cand = cand[~cand.index.isin([int(p.replace("ss_", "")) for p in placed_pids if p])]
        cand["source_player_id"] = cand.index
        # Skor
        ranked = []
        for r in cand.itertuples():
            cpid = f"ss_{int(r.source_player_id)}"
            pos_s = position_match(slot, getattr(r, "position", "") or "")
            style = float(getattr(r, "style_fit", 0.0) or 0.0) / 100.0
            perf  = float(getattr(r, "role_score", 0.0) or 0.0) / 100.0
            syn   = gnn_synergy_avg(eng, cpid, [p for p in lineup.values() if p != pid])
            syn_n = max(0.0, min(1.0, (syn or 0.0) / 1.2))
            combined = 0.20 * pos_s + 0.30 * style + 0.25 * perf + 0.25 * syn_n
            ranked.append({
                "pid": cpid,
                "name": getattr(r, "player_name", "?"),
                "team": getattr(r, "team_name", "?"),
                "position": getattr(r, "position", None),
                "combined_score": round(combined, 4),
                "components": {
                    "position": round(pos_s, 3),
                    "style_fit": round(style, 3),
                    "perf": round(perf, 3),
                    "synergy": round(syn_n, 3),
                },
                "raw": {
                    "minutes": int(getattr(r, "minutes", 0) or 0),
                    "rating":  float(getattr(r, "rating", 0) or 0),
                    "gnn_synergy_avg": round(syn, 4) if syn is not None else None,
                },
            })
        ranked.sort(key=lambda x: -x["combined_score"])
        replacements[slot] = ranked[:top_k]

    return {
        "season": season,
        "anchor_team": payload.get("anchor_team"),
        "style": payload.get("style"),
        "filters": {"min_minutes": min_minutes},
        "team_style_fit": style_pkg["team_style_fit"],
        "per_player_fit": style_pkg["per_player_fit"],
        "pairwise_synergy": pair_syn,
        "replacements": replacements,
    }
