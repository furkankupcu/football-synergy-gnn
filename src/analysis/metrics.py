"""Performans metrikleri — xG, xA, progressive pass/carry, per-90 normalizasyonu."""

import numpy as np
import pandas as pd


PER90_CANDIDATES = [
    "goals", "assists", "xg", "npxg", "xa",
    "shots", "shots_on_target",
    "progressive_passes", "progressive_carries",
    "tackles", "interceptions", "blocks",
    "touches", "carries", "passes_completed",
]


def add_per90_columns(player_stats: pd.DataFrame, minutes_col: str = "minutes") -> pd.DataFrame:
    """Convert all numeric metrics to per-90."""
    df = player_stats.copy()
    if minutes_col not in df.columns:
        return df

    minutes = pd.to_numeric(df[minutes_col], errors="coerce").replace(0, np.nan)
    scale = 90.0 / minutes

    for col in PER90_CANDIDATES:
        if col in df.columns:
            df[f"{col}_per90"] = pd.to_numeric(df[col], errors="coerce") * scale

    return df


def xg_overperformance(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Goals − xG difference (high = clinical finisher; low = poor finisher)."""
    df = player_stats.copy()
    if "goals" in df.columns and "xg" in df.columns:
        df["goals_minus_xg"] = pd.to_numeric(df["goals"], errors="coerce") - pd.to_numeric(df["xg"], errors="coerce")
    if "assists" in df.columns and "xa" in df.columns:
        df["assists_minus_xa"] = pd.to_numeric(df["assists"], errors="coerce") - pd.to_numeric(df["xa"], errors="coerce")
    return df


def top_performers(
    player_stats: pd.DataFrame,
    metric: str = "xg",
    n: int = 20,
    min_minutes: int = 450,
    position_filter: str | None = None,
) -> pd.DataFrame:
    """Rank the best players by a given metric."""
    df = player_stats.copy()
    if "minutes" in df.columns:
        df = df[pd.to_numeric(df["minutes"], errors="coerce").fillna(0) >= min_minutes]
    if position_filter and "position" in df.columns:
        df = df[df["position"].str.contains(position_filter, case=False, na=False)]
    if metric not in df.columns:
        return df.head(0)

    df = df.sort_values(metric, ascending=False)
    show = ["player", "team", "position", "minutes", metric, "goals", "assists"]
    show = [c for c in show if c in df.columns]
    return df[show].head(n).reset_index(drop=True)


def progressive_action_share(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Player shares of the team's progressive passes and carries."""
    df = player_stats.copy()
    if "team" not in df.columns:
        return df

    for col in ["progressive_passes", "progressive_carries"]:
        if col in df.columns:
            team_sum = df.groupby("team")[col].transform("sum").replace(0, np.nan)
            df[f"{col}_share_pct"] = (
                pd.to_numeric(df[col], errors="coerce") / team_sum * 100
            ).round(1)
    return df


def build_player_summary(player_stats: pd.DataFrame) -> pd.DataFrame:
    """360-degree player summary — per-90 + xG differences + share metrics."""
    df = add_per90_columns(player_stats)
    df = xg_overperformance(df)
    df = progressive_action_share(df)
    return df
