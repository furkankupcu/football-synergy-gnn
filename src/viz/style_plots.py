"""Style clustering and playing-style plots (Plotly + Matplotlib)."""

import pandas as pd
import matplotlib.pyplot as plt

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


CLUSTER_COLORS = [
    "#4fc3f7", "#f44336", "#4caf50", "#ff9800",
    "#9c27b0", "#00bcd4", "#ffeb3b", "#795548",
]

CLUSTER_NAMES = {
    0: "Pressing / High Intensity",
    1: "Passive / Counter-Attacking",
    2: "Possession",
    3: "Balanced / Mixed",
    4: "Defensive Block",
    5: "High Scoring / Open Game",
}


def style_scatter(
    clustered_teams: pd.DataFrame,
    team_col: str = "team_name",
    x_col: str = "PC1",
    y_col: str = "PC2",
    cluster_col: str = "cluster",
) -> "go.Figure | plt.Figure":
    """Scatter plot of teams in 2D PCA space, coloured by cluster."""
    df = clustered_teams.copy()
    df["cluster_name"] = df[cluster_col].map(CLUSTER_NAMES).fillna(df[cluster_col].astype(str))

    if PLOTLY_AVAILABLE:
        fig = px.scatter(
            df,
            x=x_col, y=y_col,
            color="cluster_name",
            text=team_col,
            title="Team Style Clusters (PCA)",
            template="plotly_dark",
            color_discrete_sequence=CLUSTER_COLORS,
        )
        fig.update_traces(
            textposition="top center",
            marker=dict(size=12, line=dict(width=1, color="white")),
        )
        fig.update_layout(
            font_color="white",
            plot_bgcolor="#1a1a2e",
            paper_bgcolor="#1a1a2e",
            legend_title="Oyun Stili",
        )
        return fig

    # Matplotlib fallback
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    for i, (_, row) in enumerate(df.iterrows()):
        c = CLUSTER_COLORS[int(row[cluster_col]) % len(CLUSTER_COLORS)]
        ax.scatter(row[x_col], row[y_col], color=c, s=150, zorder=5)
        ax.annotate(row[team_col], (row[x_col], row[y_col]),
                    textcoords="offset points", xytext=(5, 5), color="white", fontsize=9)
    ax.set_xlabel("PC1", color="white")
    ax.set_ylabel("PC2", color="white")
    ax.set_title("Team Style Clusters (PCA)", color="white")
    ax.tick_params(colors="white")
    return fig


def ppda_bar(
    team_stats: pd.DataFrame,
    team_col: str = "team_name",
    ppda_col: str = "ppda",
) -> "go.Figure | plt.Figure":
    """Sorted bar chart of team PPDA values."""
    df = team_stats[[team_col, ppda_col]].dropna().sort_values(ppda_col)

    if PLOTLY_AVAILABLE:
        colors = ["#f44336" if v < 8 else "#4fc3f7" if v < 12 else "#4caf50"
                  for v in df[ppda_col]]
        fig = go.Figure(go.Bar(
            x=df[ppda_col], y=df[team_col],
            orientation="h",
            marker_color=colors,
        ))
        fig.update_layout(
            title="PPDA — Team Pressing Intensity",
            xaxis_title="PPDA (lower = more pressing)",
            template="plotly_dark",
            plot_bgcolor="#1a1a2e",
            paper_bgcolor="#1a1a2e",
        )
        fig.add_vline(x=8, line_dash="dash", line_color="#ffeb3b", annotation_text="High Press Threshold")
        fig.add_vline(x=13, line_dash="dash", line_color="#ff9800", annotation_text="Low Block Threshold")
        return fig

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.barh(df[team_col], df[ppda_col], color="#4fc3f7")
    ax.axvline(8, color="#ffeb3b", linestyle="--", label="High Press")
    ax.axvline(13, color="#ff9800", linestyle="--", label="Low Block")
    ax.set_xlabel("PPDA", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    return fig


def xg_trend(
    fixtures: pd.DataFrame,
    team: str,
    n_matches: int = 10,
) -> "go.Figure | plt.Figure":
    """Plot a team's xG and xGA trend over its last n matches."""
    df = fixtures.copy()
    home_mask = df["home_team"] == team if "home_team" in df.columns else pd.Series(False, index=df.index)
    away_mask = df["away_team"] == team if "away_team" in df.columns else pd.Series(False, index=df.index)

    home = df[home_mask][["date", "home_xg", "away_xg"]].rename(
        columns={"home_xg": "xg", "away_xg": "xga"}
    )
    away = df[away_mask][["date", "away_xg", "home_xg"]].rename(
        columns={"away_xg": "xg", "home_xg": "xga"}
    )
    team_df = pd.concat([home, away]).sort_values("date").tail(n_matches)

    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=team_df["date"], y=team_df["xg"],
                                 mode="lines+markers", name="xG", line_color="#4caf50"))
        fig.add_trace(go.Scatter(x=team_df["date"], y=team_df["xga"],
                                 mode="lines+markers", name="xGA", line_color="#f44336"))
        fig.update_layout(
            title=f"{team} — xG Trend (Last {n_matches} Matches)",
            template="plotly_dark",
            plot_bgcolor="#1a1a2e",
            paper_bgcolor="#1a1a2e",
        )
        return fig

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(team_df["date"], team_df["xg"], color="#4caf50", marker="o", label="xG")
    ax.plot(team_df["date"], team_df["xga"], color="#f44336", marker="o", label="xGA")
    ax.legend()
    ax.set_title(f"{team} xG Trend")
    return fig
