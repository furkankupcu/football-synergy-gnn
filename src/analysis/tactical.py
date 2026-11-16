"""Taktiksel analiz — PPDA, pressing, formasyon, set piece."""

import numpy as np
import pandas as pd


def calculate_ppda(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    PPDA proxy: opponent pass volume is not available, so an alternative definition is used:
        PPDA_proxy = (own_passes / 3) / def_actions
    def_actions = tackles + interceptions + blocks

    True PPDA needs opponent passes (not in FBref's team season stats).
    This proxy still ranks teams correctly within a league.
    """
    df = team_stats.copy()
    def_actions = (
        df.get("tackles", 0).fillna(0)
        + df.get("interceptions", 0).fillna(0)
        + df.get("blocks", 0).fillna(0)
    )
    # On average, the team's passes per defensive action
    if "passes" in df.columns:
        df["ppda"] = (df["passes"].fillna(0) / 3) / def_actions.replace(0, np.nan)
    else:
        df["ppda"] = np.nan
    return df


def classify_pressing_style(ppda: float) -> str:
    """Classify the pressing style from the PPDA value."""
    if pd.isna(ppda):
        return "Unknown"
    if ppda < 8:
        return "High Press"
    elif ppda < 11:
        return "Mid Press"
    elif ppda < 14:
        return "Mid-Low Block"
    else:
        return "Low Block"


def defensive_intensity(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Defensive actions per 90 — an alternative metric summarising team pressing."""
    df = team_stats.copy()
    mp = df.get("matches_played", pd.Series([1] * len(df))).replace(0, np.nan)

    df["tackles_per90"] = df.get("tackles", 0) / mp
    df["interceptions_per90"] = df.get("interceptions", 0) / mp
    df["blocks_per90"] = df.get("blocks", 0) / mp
    df["def_actions_per90"] = (
        df["tackles_per90"] + df["interceptions_per90"] + df["blocks_per90"]
    )
    return df


def attack_efficiency(team_stats: pd.DataFrame) -> pd.DataFrame:
    """xG ↔ actual goals efficiency (over/under-performance)."""
    df = team_stats.copy()
    if "xg" in df.columns and "goals_for" in df.columns:
        df["goal_xg_diff"] = df["goals_for"] - df["xg"]
        df["overperforming"] = df["goal_xg_diff"] > 1
        df["underperforming"] = df["goal_xg_diff"] < -1
    return df


def build_tactical_report(team_stats: pd.DataFrame) -> pd.DataFrame:
    """PPDA + defensive intensity + attacking efficiency in one table."""
    df = calculate_ppda(team_stats)
    df = defensive_intensity(df)
    df = attack_efficiency(df)
    if "ppda" in df.columns:
        df["pressing_style"] = df["ppda"].apply(classify_pressing_style)

    keep = [
        "team_name", "league", "season",
        "matches_played", "goals_for", "goals_against",
        "xg", "xg_against", "goal_xg_diff",
        "tackles_per90", "interceptions_per90", "def_actions_per90",
        "possession", "ppda", "pressing_style",
    ]
    available = [c for c in keep if c in df.columns]
    return df[available]
