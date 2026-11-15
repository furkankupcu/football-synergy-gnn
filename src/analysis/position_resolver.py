"""Oyuncu → 11 detaylı pozisyon çözümleyicisi (web tarafı için).

GNN tarafıyla aynı mantık:
  - base_code (G/D/M/F) birincil sınıf
  - avg-position scrape verisi varsa X/Y ile alt-pozisyon türetilir
  - yoksa base_code'a default detay atanır

Sonuç: 11 kategoriden biri (GK/RB/LB/RCB/LCB/CDM/CM/CAM/RW/LW/ST).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import pandas as pd

from src import db
from src.graph.positions import coordinate_to_position


_BASE_FALLBACK: dict[str, str] = {
    "G": "GK", "D": "RCB", "M": "CM", "F": "ST",
}


@lru_cache(maxsize=4)
def get_position_map(season: Optional[str] = None) -> dict[int, str]:
    """source_player_id (int) → 11-pos string.

    Cache key sezona göre. None → tüm sezonlar bir arada (max coverage).
    """
    where = ""
    params: dict = {}
    if season:
        # Hem '2025-2026' hem '25/26' kabul et
        parts = str(season).split("-")
        if len(parts) == 2 and len(parts[0]) == 4:
            ss = f"{parts[0][2:]}/{parts[1][2:]}"
        else:
            ss = season
        where = "WHERE m.season IN (:s1, :s2)"
        params = {"s1": season, "s2": ss}

    # 1) Avg-position scrape verisi olan oyuncular: koordinat tabanlı
    sql_avg = f"""
        SELECT a.source_player_id                                  AS pid,
               MAX(a.position_code)                                    AS base_code,
               SUM(a.avg_x * a.points_count) / NULLIF(SUM(a.points_count), 0) AS x,
               SUM(a.avg_y * a.points_count) / NULLIF(SUM(a.points_count), 0) AS y,
               SUM(a.points_count)                                     AS total_pts
        FROM player_avg_positions a
        JOIN matches m ON m.match_id = a.match_id
        {where}
        GROUP BY a.source_player_id
        HAVING SUM(a.points_count) > 30
    """
    df_avg = db.read_sql(sql_avg, params)

    # 2) Avg-pos olmayanlar: en çok dakika oynadığı maçtaki base_code
    sql_base = f"""
        SELECT p.source_player_id AS pid,
               MAX(p.position)        AS base_code
        FROM player_stats p
        JOIN matches m ON m.match_id = p.match_id
        {where}
        GROUP BY p.source_player_id
    """
    df_base = db.read_sql(sql_base, params)

    pos_map: dict[int, str] = {}

    # Önce avg verisi olanlar (precise)
    for r in df_avg.itertuples():
        pos_map[int(r.pid)] = coordinate_to_position(r.x, r.y, r.base_code)

    # Sonra avg verisi olmayanlar (fallback)
    for r in df_base.itertuples():
        if int(r.pid) in pos_map:
            continue
        pos_map[int(r.pid)] = _BASE_FALLBACK.get((r.base_code or "M").upper(), "CM")

    return pos_map


def apply_resolved_position(df: pd.DataFrame,
                            id_col: str = "source_player_id",
                            pos_col: str = "position",
                            season: Optional[str] = None) -> pd.DataFrame:
    """DataFrame'in `position` kolonunu 11-pozisyon string'i ile değiştirir.

    Var olmayan eşleşmelerde mevcut değeri korur.
    """
    if df is None or df.empty:
        return df
    pmap = get_position_map(season)
    out = df.copy()
    out[pos_col] = out[id_col].astype("Int64").map(pmap).fillna(out[pos_col])
    return out


def clear_cache() -> None:
    get_position_map.cache_clear()
