"""FastAPI route'ları — SynergyHGT (GNN) inference endpoint'leri.

Mevcut rol/pozisyon sistemine dokunmadan, GNN sinerji modelini API üzerinden açar.
- /gnn/similar/{player_id}             — embedding benzeri oyuncular
- /gnn/squad-recommend                  — kadro tamamlama (POST)
- /gnn/team-map/{team_name}             — takım pozisyon haritası (avg_x/avg_y)
- /gnn/meta                             — model meta + coverage
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Query

from src import db
from src.graph.positions import FORMATIONS

router = APIRouter(prefix="/gnn", tags=["gnn"])

GRAPH_DIR = Path("data/graph")
MODELS_DIR = GRAPH_DIR / "models"


def _available_seasons() -> list[str]:
    return sorted(
        f.stem.replace("hetero_", "")
        for f in GRAPH_DIR.glob("hetero_*.pt")
        if (MODELS_DIR / f.name.replace("hetero_", "synergy_hgt_")).exists()
    )


@lru_cache(maxsize=4)
def _engine(season: str):
    """Lazy-load + cache SynergyEngine. Torch ağır; engine başına ~500MB."""
    from src.graph.recommend import SynergyEngine
    return SynergyEngine.load(season=season)


def _resolve_season(season: Optional[str]) -> str:
    seasons = _available_seasons()
    if not seasons:
        raise HTTPException(503, "Hiç eğitilmiş GNN modeli yok.")
    if season is None:
        return seasons[-1]
    if season not in seasons:
        raise HTTPException(404, f"Sezon bulunamadı: {season}. Mevcut: {seasons}")
    return season


def _season_join(season: str, alias: str = "t") -> tuple[str, dict]:
    """`JOIN matches ... WHERE <sezon>` — model sezonu dışındaki maçlar karışmasın
    (yeni sezon verisi geldikçe dakika/konum toplamları aksi halde iki sezonu toplar)."""
    from src.graph.edge_builder import _season_where
    where_sql, params = _season_where(season)
    return f"JOIN matches m ON m.match_id = {alias}.match_id {where_sql}", params


def _pid(player_id: int | str) -> str:
    s = str(player_id)
    return s if s.startswith("ss_") else f"ss_{s}"


# ---------------------------------------------------------------------------

@router.get("/meta")
def meta(season: Optional[str] = Query(None)):
    season = _resolve_season(season)
    eng = _engine(season)
    cov = db.read_sql("""
        SELECT (SELECT COUNT(DISTINCT match_id) FROM player_avg_positions) AS done,
               (SELECT COUNT(*) FROM matches) AS total
    """)
    return {
        "season": season,
        "available_seasons": _available_seasons(),
        "players": len(eng.player_ids),
        "teams": len(set(eng.teams)),
        "edges": int(eng.data["player", "teammate", "player"].edge_index.shape[1] // 2),
        "formations": list(FORMATIONS),
        "avg_pos_coverage": {
            "done": int(cov.iloc[0, 0]), "total": int(cov.iloc[0, 1]),
            "pct": round(100 * cov.iloc[0, 0] / max(cov.iloc[0, 1], 1), 1),
        },
    }


@router.get("/similar/{player_id}")
def similar(player_id: int, k: int = Query(12, ge=1, le=50),
            same_position: bool = True, season: Optional[str] = None):
    """Embedding cosine benzerliği ile en yakın oyuncular."""
    from src.graph.similar import find_similar
    season = _resolve_season(season)
    eng = _engine(season)
    pid = _pid(player_id)
    if pid not in eng.pid_to_idx:
        raise HTTPException(404, f"GNN graph'ında oyuncu yok: {pid}")
    recs = find_similar(eng, pid, k=k, same_position=same_position)
    return {
        "season": season,
        "query": {
            "player_id": pid,
            "name": eng.player_names[eng.pid_to_idx[pid]],
            "team": eng.teams[eng.pid_to_idx[pid]],
            "position": eng.positions[eng.pid_to_idx[pid]],
        },
        "results": [
            {"player_id": r.player_id, "name": r.player_name,
             "team": r.team, "position": r.position, "score": float(r.score)}
            for r in recs
        ],
    }


@router.post("/squad-recommend")
def squad_recommend(payload: dict = Body(...)):
    """
    Body:
      {
        "formation": "4-3-3",                     # FORMATIONS içinde olmalı veya slot list
        "filled": {"GK": "ss_12345", "ST": "ss_67890"},
        "top_k": 5,
        "flexible_positions": true,
        "season": "2025-2026"                     # opsiyonel
      }
    """
    season = _resolve_season(payload.get("season"))
    eng = _engine(season)
    formation = payload.get("formation")
    if not formation:
        raise HTTPException(400, "formation gerekli.")
    if isinstance(formation, str) and formation not in FORMATIONS:
        raise HTTPException(400, f"Bilinmeyen formasyon. Seçenekler: {list(FORMATIONS)}")

    raw_filled = payload.get("filled", {}) or {}
    filled = {k: _pid(v) for k, v in raw_filled.items() if v}

    # Daha fazla aday çek; min_minutes filtresinden sonra top_k'ye kırpılır
    top_k = int(payload.get("top_k", 5))
    min_minutes = int(payload.get("min_minutes", 900))
    overfetch = max(top_k * 5, 30)

    recs = eng.recommend_squad(
        formation=formation,
        filled=filled,
        top_k=overfetch,
        flexible_positions=bool(payload.get("flexible_positions", True)),
    )

    # Performans filtresi: sezon toplam dakikası min_minutes altında olanları ele
    eligible_pids: set[str] | None = None
    if min_minutes > 0:
        try:
            join_sql, params = _season_join(season)
            mins_df = db.read_sql(f"""
                SELECT 'ss_' || t.source_player_id::text AS pid,
                       COALESCE(SUM(t.minutes_played), 0) AS mins
                FROM player_stats t {join_sql}
                GROUP BY t.source_player_id
                HAVING COALESCE(SUM(t.minutes_played), 0) >= :min_minutes
            """, {**params, "min_minutes": min_minutes})
            eligible_pids = set(mins_df["pid"].tolist())
        except Exception as e:
            print(f"[squad-recommend] min_minutes filter failed: {e}")
            eligible_pids = None

    def _filter_and_trim(lst):
        if eligible_pids is not None:
            lst = [r for r in lst if r.player_id in eligible_pids]
        return lst[:top_k]

    return {
        "season": season,
        "formation": formation,
        "min_minutes": min_minutes,
        "filled": [
            {"position": pos, "player_id": pid,
             "name": eng.player_names[eng.pid_to_idx[pid]],
             "team": eng.teams[eng.pid_to_idx[pid]]}
            for pos, pid in filled.items() if pid in eng.pid_to_idx
        ],
        "recommendations": {
            pos: [
                {"player_id": r.player_id, "name": r.player_name,
                 "team": r.team, "position": r.position, "score": float(r.score)}
                for r in _filter_and_trim(lst)
            ]
            for pos, lst in recs.items()
        },
    }


@router.post("/pair-scores")
def pair_scores(payload: dict = Body(...)):
    """Verilen oyuncu çiftleri için GNN sinerji skoru.

    Body: {"pairs": [["ss_x","ss_y"], ...], "season": "2025-2026"}
    Response: [{"a": "ss_x", "b": "ss_y", "score": 0.81}, ...]
    """
    import torch
    season = _resolve_season(payload.get("season"))
    eng = _engine(season)
    pairs = payload.get("pairs") or []
    if not isinstance(pairs, list):
        raise HTTPException(400, "pairs list olmalı.")
    out = []
    for p in pairs:
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            continue
        a, b = str(p[0]), str(p[1])
        ai = eng.pid_to_idx.get(a)
        bi = eng.pid_to_idx.get(b)
        if ai is None or bi is None:
            out.append({"a": a, "b": b, "score": None})
            continue
        with torch.no_grad():
            src = torch.tensor([ai], dtype=torch.long, device=eng.device)
            dst = torch.tensor([bi], dtype=torch.long, device=eng.device)
            s = eng.model.predict_edge(eng._emb, src, dst).item()
        out.append({"a": a, "b": b, "score": float(s)})
    return {"season": season, "pairs": out}


@router.get("/player/{player_id}/raw-positions")
def player_raw_positions(player_id: int, season: Optional[str] = None):
    """Maç başına ham avg-position noktaları (per-player heatmap için)."""
    season = _resolve_season(season)
    join_sql, params = _season_join(season)
    raw = db.read_sql(f"""
        SELECT t.match_id, t.avg_x AS x, t.avg_y AS y, COALESCE(t.points_count, 0) AS points
        FROM player_avg_positions t {join_sql}
          AND t.source_player_id = :pid
          AND t.avg_x IS NOT NULL AND t.avg_y IS NOT NULL
    """, {**params, "pid": int(player_id)})
    return {
        "season": season,
        "player_id": f"ss_{player_id}",
        "count": int(len(raw)),
        "points": [
            {"match_id": str(r.match_id), "x": float(r.x), "y": float(r.y), "weight": int(r.points or 0)}
            for r in raw.itertuples()
        ],
    }


@router.get("/team-map/{team_name}")
def team_map(team_name: str, season: Optional[str] = None):
    """Avg-position scrape verisi olan oyuncuların X/Y haritası."""
    season = _resolve_season(season)
    eng = _engine(season)
    ids = [pid.removeprefix("ss_") for pid, t in zip(eng.player_ids, eng.teams)
           if t == team_name and pid.startswith("ss_") and pid[3:].isdigit()]
    if not ids:
        raise HTTPException(404, f"Takım yok veya boş: {team_name}")
    ids_csv = ",".join(ids)  # yalnız rakam (yukarıda isdigit) → enjeksiyon yok
    join_sql, params = _season_join(season)
    raw = db.read_sql(f"""
        SELECT 'ss_' || t.source_player_id::text AS player_id,
               MAX(t.player_name)                    AS name,
               SUM(t.avg_x * t.points_count) / NULLIF(SUM(t.points_count), 0) AS x,
               SUM(t.avg_y * t.points_count) / NULLIF(SUM(t.points_count), 0) AS y,
               SUM(t.points_count)                   AS points_count
        FROM player_avg_positions t {join_sql}
          AND t.source_player_id IN ({ids_csv})
        GROUP BY t.source_player_id
        HAVING SUM(t.points_count) > 30
        ORDER BY points_count DESC
    """, params)
    out = []
    for r in raw.itertuples():
        idx = eng.pid_to_idx.get(r.player_id)
        out.append({
            "player_id": r.player_id, "name": r.name,
            "x": float(r.x), "y": float(r.y), "points": int(r.points_count),
            "position": eng.positions[idx] if idx is not None else None,
        })
    return {"season": season, "team": team_name, "players": out}


@router.get("/player/{player_id}/avg-position")
def player_avg_position(player_id: int, season: Optional[str] = None):
    """Tek oyuncunun ağırlıklı X/Y ortalaması (modal mini saha için)."""
    season = _resolve_season(season)
    join_sql, params = _season_join(season)
    raw = db.read_sql(f"""
        SELECT 'ss_' || t.source_player_id::text AS player_id,
               MAX(t.player_name) AS name,
               SUM(t.avg_x * t.points_count) / NULLIF(SUM(t.points_count), 0) AS x,
               SUM(t.avg_y * t.points_count) / NULLIF(SUM(t.points_count), 0) AS y,
               SUM(t.points_count) AS points_count,
               COUNT(*)          AS matches
        FROM player_avg_positions t {join_sql}
          AND t.source_player_id = :pid
        GROUP BY t.source_player_id
    """, {**params, "pid": int(player_id)})
    if raw.empty:
        return {"season": season, "player_id": f"ss_{player_id}", "x": None, "y": None, "points": 0, "matches": 0}
    r = raw.iloc[0]
    return {
        "season": season,
        "player_id": r["player_id"],
        "name": r["name"],
        "x": float(r["x"]) if r["x"] is not None else None,
        "y": float(r["y"]) if r["y"] is not None else None,
        "points": int(r["points_count"] or 0),
        "matches": int(r["matches"] or 0),
    }


@router.get("/team-roster/{team_name}")
def team_roster(team_name: str, season: Optional[str] = None):
    """Takımın GNN graf'ındaki oyuncuları — pozisyon başına dakika DESC sıralı.

    UI 'auto-fill' için kullanır: her formasyon slot'una en çok dakika oynayan
    o pozisyondaki oyuncuyu yerleştirir.
    """
    season = _resolve_season(season)
    eng = _engine(season)
    ids = [pid for pid, t in zip(eng.player_ids, eng.teams) if t == team_name]
    if not ids:
        raise HTTPException(404, f"Takım yok veya boş: {team_name}")

    numeric_ids = [pid.removeprefix("ss_") for pid in ids
                   if pid.startswith("ss_") and pid[3:].isdigit()]
    minutes_map: dict[str, int] = {}
    if numeric_ids:
        ids_csv = ",".join(numeric_ids)  # yalnız rakam → enjeksiyon yok
        join_sql, params = _season_join(season)
        m_df = db.read_sql(f"""
            SELECT 'ss_' || t.source_player_id::text AS player_id,
                   SUM(COALESCE(t.minutes_played, 0))::int AS minutes
            FROM player_stats t {join_sql}
              AND t.source_player_id IN ({ids_csv})
            GROUP BY t.source_player_id
        """, params)
        minutes_map = {r.player_id: int(r.minutes) for r in m_df.itertuples()}

    players = []
    for pid in ids:
        idx = eng.pid_to_idx[pid]
        players.append({
            "player_id": pid,
            "name": eng.player_names[idx],
            "position": eng.positions[idx],
            "minutes": minutes_map.get(pid, 0),
        })
    players.sort(key=lambda p: -p["minutes"])
    return {"season": season, "team": team_name, "players": players}


@router.get("/teams")
def teams(q: Optional[str] = Query(None, description="İsim araması (kısmi)"),
          season: Optional[str] = None, limit: int = Query(30, ge=1, le=2000)):
    """GNN graph'taki benzersiz takım listesi (autocomplete için)."""
    season = _resolve_season(season)
    eng = _engine(season)

    def _fold(s: str) -> str:
        # Türkçe + aksanlı karakterleri ASCII'ye indir, lower
        repl = str.maketrans({
            "ç": "c", "Ç": "c", "ğ": "g", "Ğ": "g", "ı": "i", "İ": "i",
            "ö": "o", "Ö": "o", "ş": "s", "Ş": "s", "ü": "u", "Ü": "u",
            "â": "a", "î": "i", "û": "u", "é": "e", "á": "a", "í": "i", "ó": "o", "ú": "u",
        })
        return s.translate(repl).lower()

    q_fold = _fold(q) if q else None
    seen: dict[str, int] = {}
    for t in eng.teams:
        if not t or t == "?":
            continue
        if q_fold and q_fold not in _fold(t):
            continue
        seen[t] = seen.get(t, 0) + 1
        if len(seen) >= limit and not q_fold:
            break
    out = [{"team": t, "players": n} for t, n in sorted(seen.items(), key=lambda kv: -kv[1])]
    return {"season": season, "count": len(out), "teams": out[:limit]}


@router.get("/players")
def players(team: Optional[str] = None, position: Optional[str] = None,
            q: Optional[str] = Query(None, description="İsim araması"),
            season: Optional[str] = None, limit: int = Query(200, ge=1, le=20000)):
    """GNN graph içindeki oyuncu listesi (UI dropdown'ları için)."""
    season = _resolve_season(season)
    eng = _engine(season)
    out = []
    def _fold(s: str) -> str:
        repl = str.maketrans({
            "ç": "c", "Ç": "c", "ğ": "g", "Ğ": "g", "ı": "i", "İ": "i",
            "ö": "o", "Ö": "o", "ş": "s", "Ş": "s", "ü": "u", "Ü": "u",
            "â": "a", "î": "i", "û": "u", "é": "e", "á": "a", "í": "i", "ó": "o", "ú": "u",
        })
        return (s or "").translate(repl).lower()
    q_fold = _fold(q) if q else None
    for i in range(len(eng.player_ids)):
        if team and eng.teams[i] != team:
            continue
        if position and eng.positions[i] != position:
            continue
        nm = eng.player_names[i]
        if q_fold and q_fold not in _fold(nm):
            continue
        out.append({
            "player_id": eng.player_ids[i], "name": nm,
            "team": eng.teams[i], "position": eng.positions[i],
        })
        if len(out) >= limit:
            break
    return {"season": season, "count": len(out), "players": out}
