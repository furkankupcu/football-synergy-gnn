"""Radar charts — player and team comparisons (mplsoccer)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



PLAYER_RADAR_PARAMS = [
    "goals", "assists", "xg", "xa",
    "progressive_passes", "progressive_carries",
    "tackles", "interceptions",
]

PLAYER_RADAR_LABELS = [
    "Goals", "Assists", "xG", "xA",
    "Prog. Passes", "Prog. Carries",
    "Tackles", "Interceptions",
]

TEAM_RADAR_PARAMS = [
    "xg", "xg_against", "possession",
    "progressive_passes", "progressive_carries",
    "shots", "shots_on_target", "passes_pct",
    "tackles", "interceptions",
]

TEAM_RADAR_LABELS = [
    "xG", "xGA", "Possession",
    "Prog. Passes", "Prog. Carries",
    "Shots", "Shots on Target", "Pass %",
    "Tackles", "Interceptions",
]


def _percentile_radar(
    values: np.ndarray,
    params: list[str],
    labels: list[str],
    title: str,
    color: str = "#4fc3f7",
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """Draw a basic percentile radar chart (fallback without mplsoccer)."""
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    fig.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    N = len(params)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values.tolist()
    angles += angles[:1]
    values_plot += values_plot[:1]

    ax.plot(angles, values_plot, color=color, linewidth=2)
    ax.fill(angles, values_plot, color=color, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="white", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="#888888", fontsize=7)
    ax.yaxis.grid(True, color="#333355", linestyle="--", alpha=0.5)
    ax.xaxis.grid(True, color="#333355", linestyle="--", alpha=0.3)
    ax.set_title(title, color="white", fontsize=14, pad=20, y=1.1)
    ax.tick_params(colors="white")
    return fig


def player_radar(
    player_stats: pd.DataFrame,
    player_name: str,
    params: list[str] | None = None,
    labels: list[str] | None = None,
    color: str = "#4fc3f7",
) -> plt.Figure:
    """Percentile radar chart for a player."""
    params = params or PLAYER_RADAR_PARAMS
    labels = labels or PLAYER_RADAR_LABELS

    available = [p for p in params if p in player_stats.columns]
    labels = [labels[i] for i, p in enumerate(params) if p in player_stats.columns]
    params = available

    if not params:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
        return fig

    X = player_stats[params].apply(pd.to_numeric, errors="coerce").fillna(0)
    percentiles = X.rank(pct=True) * 100

    mask = player_stats["player"] == player_name
    if not mask.any():
        raise ValueError(f"Player not found: {player_name}")

    values = percentiles[mask].iloc[0].values
    return _percentile_radar(values, params, labels, player_name, color)


def team_radar(
    team_stats: pd.DataFrame,
    team_name: str,
    name_col: str = "team_name",
    params: list[str] | None = None,
    labels: list[str] | None = None,
    color: str = "#ff9800",
) -> plt.Figure:
    """Percentile radar chart for a team."""
    params = params or TEAM_RADAR_PARAMS
    labels = labels or TEAM_RADAR_LABELS

    available = [p for p in params if p in team_stats.columns]
    labels = [labels[i] for i, p in enumerate(params) if p in team_stats.columns]
    params = available

    if not params:
        raise ValueError("Not enough team metrics.")

    if name_col in team_stats.columns:
        mask = team_stats[name_col] == team_name
    else:
        mask = team_stats.index == team_name

    if not mask.any():
        raise ValueError(f"Team not found: {team_name}")

    X = team_stats[params].apply(pd.to_numeric, errors="coerce").fillna(0)
    percentiles = X.rank(pct=True) * 100
    values = percentiles[mask.values if hasattr(mask, "values") else mask].iloc[0].values
    return _percentile_radar(values, params, labels, team_name, color)


def comparison_radar(
    stats: pd.DataFrame,
    entities: list[str],
    entity_col: str = "player",
    params: list[str] | None = None,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
) -> plt.Figure:
    """Compare two or more players/teams on one radar chart."""
    params = params or PLAYER_RADAR_PARAMS
    labels = labels or PLAYER_RADAR_LABELS
    colors = colors or ["#4fc3f7", "#f44336", "#4caf50", "#ff9800"]

    available = [p for p in params if p in stats.columns]
    labels = [labels[i] for i, p in enumerate(params) if p in stats.columns]
    params = available

    X = stats[params].apply(pd.to_numeric, errors="coerce").fillna(0)
    percentiles = X.rank(pct=True) * 100

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    fig.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    N = len(params)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for entity, color in zip(entities, colors):
        mask = stats[entity_col] == entity
        if not mask.any():
            continue
        values = percentiles[mask].iloc[0].values.tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=entity)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="white", fontsize=9)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color="#333355", linestyle="--", alpha=0.5)
    ax.legend(facecolor="#1a1a2e", edgecolor="white", labelcolor="white", loc="upper right")
    ax.set_title("Comparison", color="white", fontsize=14, pad=20, y=1.1)
    return fig
