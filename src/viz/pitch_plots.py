"""Pitch visualisations — shot map, pass map, heatmap (mplsoccer)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    from mplsoccer import Pitch, VerticalPitch
    MPLSOCCER_AVAILABLE = True
except ImportError:
    MPLSOCCER_AVAILABLE = False


def _check_mplsoccer():
    if not MPLSOCCER_AVAILABLE:
        raise ImportError("mplsoccer is not installed: pip install mplsoccer")


def shot_map(
    shots: pd.DataFrame,
    team: str | None = None,
    title: str = "Shot Map",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Draw a shot map with xG bubbles.
    shots: must contain the columns ['x','y','xg','outcome','team'] (StatsBomb coordinates: 0-120, 0-80).
    """
    _check_mplsoccer()

    if team and "team" in shots.columns:
        shots = shots[shots["team"] == team]

    pitch = VerticalPitch(
        pitch_type="statsbomb",
        pitch_color="#1a1a2e",
        line_color="#ffffff",
        half=True,
    )
    fig, ax = pitch.draw(figsize=figsize)
    fig.set_facecolor("#1a1a2e")

    goals = shots[shots.get("outcome", pd.Series(["miss"] * len(shots))).str.lower() == "goal"] if "outcome" in shots.columns else pd.DataFrame()
    non_goals = shots[shots.get("outcome", pd.Series(["miss"] * len(shots))).str.lower() != "goal"] if "outcome" in shots.columns else shots

    xg_col = shots["xg"] if "xg" in shots.columns else pd.Series([0.1] * len(shots))

    if not non_goals.empty:
        scatter_size = (non_goals.get("xg", 0.1) * 1000).clip(50, 1500)
        pitch.scatter(
            non_goals["x"], non_goals["y"],
            s=scatter_size,
            c="#4fc3f7", edgecolors="#ffffff", linewidths=0.5,
            alpha=0.7, ax=ax, label="Shot",
        )

    if not goals.empty:
        scatter_size = (goals.get("xg", 0.3) * 1000).clip(100, 2000)
        pitch.scatter(
            goals["x"], goals["y"],
            s=scatter_size,
            c="#f44336", edgecolors="#ffffff", linewidths=1,
            alpha=0.9, ax=ax, label="Goal", marker="*", zorder=5,
        )

    ax.set_title(f"{team or ''} {title}", color="white", fontsize=14, pad=15)
    ax.legend(facecolor="#1a1a2e", edgecolor="white", labelcolor="white", loc="lower left")

    total_xg = xg_col.sum()
    total_shots = len(shots)
    ax.text(0.5, 0.02, f"Total shots: {total_shots} | Total xG: {total_xg:.2f}",
            transform=ax.transAxes, ha="center", color="#cccccc", fontsize=10)
    return fig


def pass_map(
    passes: pd.DataFrame,
    team: str | None = None,
    progressive_only: bool = False,
    title: str = "Pass Map",
    figsize: tuple = (14, 9),
) -> plt.Figure:
    """
    Draw a pass map. passes: columns ['x','y','end_x','end_y','progressive','team'].
    """
    _check_mplsoccer()

    if team and "team" in passes.columns:
        passes = passes[passes["team"] == team]
    if progressive_only and "progressive" in passes.columns:
        passes = passes[passes["progressive"]]

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#1a1a2e",
        line_color="#ffffff",
    )
    fig, ax = pitch.draw(figsize=figsize)
    fig.set_facecolor("#1a1a2e")

    if passes.empty:
        ax.set_title("No data", color="white")
        return fig

    pitch.arrows(
        passes["x"], passes["y"],
        passes["end_x"], passes["end_y"],
        width=1.5, headwidth=8, headlength=5,
        color="#4fc3f7", alpha=0.5, ax=ax,
    )
    ax.set_title(f"{team or ''} {title}", color="white", fontsize=14, pad=15)
    return fig


def heatmap(
    positions: pd.DataFrame,
    team: str | None = None,
    title: str = "Heatmap",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Heatmap of player positions or touch density.
    positions: must contain the columns ['x','y'].
    """
    _check_mplsoccer()

    if team and "team" in positions.columns:
        positions = positions[positions["team"] == team]

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#1a1a2e",
        line_color="#ffffff",
    )
    fig, ax = pitch.draw(figsize=figsize)
    fig.set_facecolor("#1a1a2e")

    if positions.empty:
        ax.set_title("No data", color="white")
        return fig

    pitch.kdeplot(
        positions["x"], positions["y"],
        ax=ax, fill=True, levels=100,
        cmap="hot", alpha=0.7,
    )
    ax.set_title(f"{team or ''} {title}", color="white", fontsize=14, pad=15)
    return fig
