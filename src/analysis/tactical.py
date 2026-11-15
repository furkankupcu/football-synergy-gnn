"""Taktiksel analiz — PPDA, pressing, formasyon, set piece."""

import numpy as np
import pandas as pd


def calculate_ppda(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    PPDA proxy: Rakip pas hacmi yok, bu yüzden alternatif tanım kullanılır:
        PPDA_proxy = (own_passes / 3) / def_actions
    def_actions = tackles + interceptions + blocks

    Gerçek PPDA için rakip pasları gerekir (FBref'in team season stats'ında yok).
    Bu proxy lig içi sıralamada doğru bilgi verir.
    """
    df = team_stats.copy()
    def_actions = (
        df.get("tackles", 0).fillna(0)
        + df.get("interceptions", 0).fillna(0)
        + df.get("blocks", 0).fillna(0)
    )
    # Ortalama olarak takımın yaptığı pas / aksiyon
    if "passes" in df.columns:
        df["ppda"] = (df["passes"].fillna(0) / 3) / def_actions.replace(0, np.nan)
    else:
        df["ppda"] = np.nan
    return df


def classify_pressing_style(ppda: float) -> str:
    """PPDA değerine göre pressing stilini sınıflandırır."""
    if pd.isna(ppda):
        return "Bilinmiyor"
    if ppda < 8:
        return "Yüksek Pres"
    elif ppda < 11:
        return "Orta Pres"
    elif ppda < 14:
        return "Orta-Düşük Blok"
    else:
        return "Düşük Blok"


def defensive_intensity(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Defensive actions per 90 üretir — takım pressingini özetleyen alternatif metrik."""
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
    """xG ↔ gerçek gol verimliliği (overperform / underperform tespiti)."""
    df = team_stats.copy()
    if "xg" in df.columns and "goals_for" in df.columns:
        df["goal_xg_diff"] = df["goals_for"] - df["xg"]
        df["overperforming"] = df["goal_xg_diff"] > 1
        df["underperforming"] = df["goal_xg_diff"] < -1
    return df


def build_tactical_report(team_stats: pd.DataFrame) -> pd.DataFrame:
    """PPDA + savunma yoğunluğu + atak verimliliğini tek tabloda özetler."""
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
