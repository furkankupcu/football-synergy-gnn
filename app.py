"""Streamlit UI — SynergyHGT inference (pitch visualisation + dashboard).

Run:
    .venv/Scripts/streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from mplsoccer import Pitch
import matplotlib.pyplot as plt

from src import db
from src.graph.positions import FORMATIONS
from src.graph.recommend import SynergyEngine
from src.graph.similar import find_similar

GRAPH_DIR = Path("data/graph")
MODELS_DIR = GRAPH_DIR / "models"

PITCH_BG = "#0b1220"
PITCH_LINE = "#5b6478"
ACCENT = "#22d3ee"
WARN = "#fbbf24"
TEAM_DOT = "#f97316"


# Pitch X/Y per position (mplsoccer: pitch 120×80; X = forward, Y = left-right)
POS_PITCH_XY: dict[str, tuple[float, float]] = {
    "GK":  (10, 40),
    "RB":  (35, 12),    "RCB": (32, 30),  "LCB": (32, 50),  "LB":  (35, 68),
    "CDM": (52, 40),    "CM":  (62, 40),  "CAM": (78, 40),
    "RW":  (80, 14),    "LW":  (80, 66),
    "ST":  (100, 40),
}


# ----- Streamlit page config + CSS -------------------------------------------
st.set_page_config(
    page_title="Football Synergy GNN",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.2rem; max-width: 1300px; }
  h1, h2, h3 { letter-spacing: -0.5px; }
  /* Metric cards */
  [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
  [data-testid="stMetricLabel"] { color: #94a3b8; }
  /* DataFrame */
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
  /* Tabs */
  div[data-baseweb="tab-list"] { gap: 4px; }
  button[data-baseweb="tab"] {
    background: #131e36; border-radius: 8px 8px 0 0; padding: 6px 14px;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    background: #22d3ee20; color: #22d3ee;
  }
  /* Sidebar */
  section[data-testid="stSidebar"] { background: #0d1730; }
  /* Cards */
  .card {
    background: #131e36;
    border: 1px solid #1f2a44;
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
  }
  .pill {
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 11px; background: #22d3ee20; color: #22d3ee;
    margin-right: 6px;
  }
</style>
""", unsafe_allow_html=True)


# ----- Helpers ---------------------------------------------------------------
def _available_seasons() -> list[str]:
    return sorted(
        f.stem.replace("hetero_", "")
        for f in GRAPH_DIR.glob("hetero_*.pt")
        if (MODELS_DIR / f.name.replace("hetero_", "synergy_hgt_")).exists()
    )


@st.cache_resource(show_spinner="Loading model...")
def load_engine(season: str) -> SynergyEngine:
    return SynergyEngine.load(season=season)


def players_df(eng: SynergyEngine) -> pd.DataFrame:
    return pd.DataFrame({
        "player_id":   eng.player_ids,
        "player_name": eng.player_names,
        "position":    eng.positions,
        "team":        eng.teams,
    })


@st.cache_data(show_spinner=False)
def avg_pos_coverage(_season: str) -> dict:
    df = db.read_sql("""
        SELECT
            (SELECT COUNT(DISTINCT match_id) FROM player_avg_positions) AS done,
            (SELECT COUNT(*) FROM matches) AS total
    """)
    done, total = int(df.iloc[0, 0]), int(df.iloc[0, 1])
    return {"done": done, "total": total, "pct": (done / total * 100) if total else 0}


def render_pitch(highlight: list[dict] | None = None,
                 candidates: list[dict] | None = None,
                 title: str = "") -> plt.Figure:
    """highlight: filled anchor players [{name, pos}]
       candidates: suggestion header per position [{pos, n}]"""
    pitch = Pitch(pitch_type="statsbomb", pitch_color=PITCH_BG,
                  line_color=PITCH_LINE, linewidth=1.2)
    fig, ax = pitch.draw(figsize=(8, 5.4))
    fig.patch.set_facecolor(PITCH_BG)
    ax.set_facecolor(PITCH_BG)

    # Empty slot markers — every position in the formation
    drawn = set()
    if candidates:
        for c in candidates:
            xy = POS_PITCH_XY.get(c["pos"])
            if not xy or c["pos"] in drawn:
                continue
            drawn.add(c["pos"])
            ax.scatter(*xy, s=720, marker="o", c="#1f2a44",
                       edgecolors=ACCENT, linewidth=1.5, zorder=2)
            ax.text(xy[0], xy[1], c["pos"], ha="center", va="center",
                    color=ACCENT, fontsize=9, fontweight="bold", zorder=3)
            if c.get("count"):
                ax.text(xy[0], xy[1] - 7, f"{c['count']} candidates",
                        ha="center", color="#94a3b8", fontsize=7)

    # Anchor (filled) players
    if highlight:
        for h in highlight:
            xy = POS_PITCH_XY.get(h["pos"])
            if not xy:
                continue
            ax.scatter(*xy, s=720, marker="o", c=TEAM_DOT,
                       edgecolors="white", linewidth=1.5, zorder=4)
            label = h["name"].split()[-1][:9]
            ax.text(xy[0], xy[1], label, ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold", zorder=5)

    if title:
        ax.set_title(title, color=ACCENT, fontsize=11, pad=10)
    return fig


# ----- Pages -----------------------------------------------------------------
def page_overview(eng: SynergyEngine, season: str):
    st.subheader("📊 Overview")
    df = players_df(eng)
    cov = avg_pos_coverage(season)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Season", season)
    c2.metric("Players", f"{len(df):,}")
    c3.metric("Teams", f"{df['team'].nunique():,}")
    c4.metric("Edges (synergy)", f"{eng.data['player','teammate','player'].edge_index.shape[1]//2:,}")

    metrics_path = MODELS_DIR / f"metrics_{season}.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        c5.metric("Test ρ (Spearman)", f"{m.get('test_rho', 0):.3f}")

    # Avg_pos coverage banner
    if cov["pct"] < 100:
        st.warning(
            f"⏳ **Avg-position scrape in progress: {cov['done']:,}/{cov['total']:,} "
            f"matches ({cov['pct']:.0f}%)** — until it finishes, some teams show coarse "
            f"positions (CM/RCB) instead of detailed ones (CAM/CDM/LB/RB).",
            icon="⚠️",
        )

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("##### Position distribution")
        st.bar_chart(df["position"].value_counts())
    with col2:
        st.markdown("##### Players per league")
        # Fetch league info from the DB
        league_df = db.read_sql("""
            SELECT m.league, COUNT(DISTINCT p.source_player_id) AS players
            FROM player_stats p
            JOIN matches m ON m.match_id = p.match_id
            WHERE m.season IN ('25/26','2025-2026')
            GROUP BY m.league ORDER BY players DESC
        """)
        st.bar_chart(league_df.set_index("league"))


def page_similar(eng: SynergyEngine):
    st.subheader("🔍 Similar Players")
    st.caption("Embedding cosine similarity — for transfer scouting.")
    df = players_df(eng)

    cols = st.columns([1, 1, 1, 2])
    with cols[0]:
        team = st.selectbox("Team filter", ["(all)"] + sorted(df["team"].unique()))
    with cols[1]:
        same_pos = st.toggle("Same position", True)
    with cols[2]:
        k = st.slider("Results", 5, 30, 12)
    with cols[3]:
        query = st.text_input("🔎 Search player",
                              placeholder="Type a name (e.g. Aktürkoğlu, Saka)")

    pool = df if team == "(all)" else df[df["team"] == team]
    if query.strip():
        pool = pool[pool["player_name"].str.contains(query.strip(), case=False, na=False)]
    pool = pool.sort_values("player_name")

    if pool.empty:
        st.warning("No players with these filters.")
        return

    label_to_id = {f"{r.player_name} · {r.team} ({r.position})": r.player_id
                   for r in pool.itertuples()}
    choice = st.selectbox(f"Player ({len(label_to_id)} results)", list(label_to_id))
    pid = label_to_id[choice]

    if st.button("Benzerleri bul", type="primary", use_container_width=True):
        recs = find_similar(eng, pid, k=k, same_position=same_pos)
        if not recs:
            st.info("No results.")
            return
        out = pd.DataFrame([{
            "Score": round(r.score, 4),
            "Player": r.player_name,
            "Position": r.position,
            "Team": r.team,
        } for r in recs])
        st.dataframe(
            out,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Similarity", min_value=0.0, max_value=1.0, format="%.4f"),
            },
        )


def page_squad(eng: SynergyEngine):
    st.subheader("🧩 Squad Completion")
    st.caption("Formation + fixed players → the model suggests synergy-maximising players for the remaining slots. "
               "You can pick **any** player of the team as an anchor and place them in **any position**.")

    df = players_df(eng)
    top_row = st.columns([1, 1, 1, 1])
    with top_row[0]:
        formation = st.selectbox("Formation", list(FORMATIONS), index=0)
    with top_row[1]:
        team = st.selectbox("Anchor team", ["(none)"] + sorted(df["team"].unique()))
    with top_row[2]:
        top_k = st.slider("Suggestions per position", 1, 10, 5)
    with top_row[3]:
        flexible = st.toggle("Flexible positions", value=True,
                             help="When on, a slot also gets candidates from the nearby position family "
                                  "(e.g. CM/RW/LW/ST for a CAM slot).")

    slots = FORMATIONS[formation]
    pos_counts: dict[str, int] = {}
    for s in slots:
        pos_counts[s] = pos_counts.get(s, 0) + 1

    left, right = st.columns([3, 4])

    # LEFT: flexible anchor picker — choose from ALL players of the team for every slot
    filled: dict[str, str] = {}
    with left:
        st.markdown("##### Anchor players (flexible)")
        if team == "(none)":
            st.info("Select a team (top left) to pick anchors.")
        else:
            team_pool = df[df["team"] == team].sort_values("player_name")
            if team_pool.empty:
                st.warning(f"{team} squad is empty.")
            else:
                slot_search = st.text_input(
                    "🔎 Search player (filters every slot)",
                    placeholder="Type a name (e.g. Saka, Akturk)",
                    key="slot_search",
                )
                already_chosen: set[str] = set()
                for pos in sorted(pos_counts):
                    suggested = team_pool[team_pool["position"] == pos]
                    others = team_pool[team_pool["position"] != pos]
                    if slot_search.strip():
                        q = slot_search.strip()
                        suggested = suggested[suggested["player_name"].str.contains(q, case=False, na=False)]
                        others = others[others["player_name"].str.contains(q, case=False, na=False)]
                    ordered = {"(empty)": None}
                    for r in suggested.itertuples():
                        ordered[f"⭐ {r.player_name} [{r.position}]"] = r.player_id
                    for r in others.itertuples():
                        ordered[f"{r.player_name} [{r.position}]"] = r.player_id
                    sel = st.selectbox(f"{pos}", list(ordered), key=f"a_{pos}")
                    pid = ordered[sel]
                    if pid is not None and pid not in already_chosen:
                        filled[pos] = pid
                        already_chosen.add(pid)
                st.caption("⭐ = the actual player for this position. You can also place other players "
                           "in any slot.")

    # RIGHT: pitch + results
    with right:
        st.markdown("##### Pitch layout")
        highlight = []
        for pos, pid in filled.items():
            name = df[df["player_id"] == pid]["player_name"].iloc[0]
            highlight.append({"name": name, "pos": pos})
        cand_info = [{"pos": p, "count": n} for p, n in pos_counts.items()
                     if p not in filled]
        fig = render_pitch(highlight=highlight, candidates=cand_info,
                           title=f"{formation} — {team if team!='(none)' else 'squad layout'}")
        st.pyplot(fig, use_container_width=True)

    st.divider()
    if st.button("🔮 Compute suggestions", type="primary", use_container_width=True):
        with st.spinner("Computing synergy scores..."):
            recs = eng.recommend_squad(
                formation=formation, filled=filled,
                top_k=top_k, flexible_positions=flexible,
            )
        any_rec = any(recs.values())
        if not any_rec:
            st.warning("No suggestions produced — no players in these positions, or all are anchors.")
            return

        # Show results in tabs
        empty_positions = [p for p in pos_counts if p not in filled]
        tabs = st.tabs(empty_positions or ["(all filled)"])
        for tab, pos in zip(tabs, empty_positions):
            with tab:
                lst = recs.get(pos, [])
                if not lst:
                    st.info(f"No candidates for {pos}.")
                    continue
                out = pd.DataFrame([{
                    "Score": float(r.score),
                    "Player": r.player_name,
                    "Team": r.team,
                } for r in lst])
                st.dataframe(
                    out,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            "Synergy",
                            min_value=float(out["Score"].min()),
                            max_value=float(out["Score"].max()),
                            format="%.3f",
                        ),
                    },
                )


def page_team_map(eng: SynergyEngine):
    st.subheader("🗺️ Team Position Map")
    st.caption("Real on-pitch X/Y position for players with avg-position data.")

    df = players_df(eng)
    team = st.selectbox("Team", sorted(df["team"].unique()))
    df_team = df[df["team"] == team]

    # fetch avg_x/avg_y
    ids = [pid.removeprefix("ss_") for pid in df_team["player_id"]]
    if not ids:
        st.warning("No team.")
        return
    int_ids = [int(x) for x in ids if str(x).isdigit()]
    if not int_ids:
        st.warning("No valid player IDs.")
        return
    ids_csv = ",".join(str(i) for i in int_ids)
    raw = db.read_sql(f"""
        SELECT
            'ss_' || source_player_id::text AS player_id,
            SUM(avg_x * points_count) / NULLIF(SUM(points_count),0) AS x,
            SUM(avg_y * points_count) / NULLIF(SUM(points_count),0) AS y,
            SUM(points_count) AS pts,
            MAX(player_name) AS name
        FROM player_avg_positions
        WHERE source_player_id IN ({ids_csv})
        GROUP BY source_player_id
        HAVING SUM(points_count) > 30
        ORDER BY pts DESC
    """)
    if raw.empty:
        st.warning(
            f"No avg-position data for {team} yet. "
            "It will appear here once the avg-position scrape finishes."
        )
        return

    # Convert coordinates to the mplsoccer pitch
    # Coordinates: X 0 = own goal, 100 = opponent; Y 0 = right, 100 = left
    # Mplsoccer (statsbomb): X 0 = defensive goal, 120 = attack; Y 0 = bottom, 80 = top
    raw["pitch_x"] = raw["x"] / 100 * 120
    raw["pitch_y"] = (100 - raw["y"]) / 100 * 80  # Y flip

    pitch = Pitch(pitch_type="statsbomb", pitch_color=PITCH_BG,
                  line_color=PITCH_LINE, linewidth=1.2)
    fig, ax = pitch.draw(figsize=(9, 6))
    fig.patch.set_facecolor(PITCH_BG)
    sizes = (raw["pts"] / raw["pts"].max() * 600 + 80).values
    ax.scatter(raw["pitch_x"], raw["pitch_y"], s=sizes,
               c=TEAM_DOT, edgecolors="white", linewidth=1, alpha=0.85, zorder=3)
    for r in raw.itertuples():
        label = r.name.split()[-1][:10]
        ax.text(r.pitch_x, r.pitch_y + 2.5, label,
                ha="center", color="white", fontsize=8, zorder=4)
    ax.set_title(f"{team} — average on-pitch position (avg-position scrape)",
                 color=ACCENT, fontsize=11, pad=8)
    st.pyplot(fig, use_container_width=True)


# ----- Main ------------------------------------------------------------------
def main():
    seasons = _available_seasons()
    if not seasons:
        st.error("No trained model. Run `python -m src.graph.train --season <ss>` first.")
        return

    with st.sidebar:
        st.markdown("### ⚽ Football Synergy")
        st.caption("GNN-based player synergy model")
        season = st.selectbox("Season", seasons, index=len(seasons) - 1)
        st.divider()
        page = st.radio(
            "Page",
            ["📊 Overview", "🔍 Similar Players",
             "🧩 Squad Completion", "🗺️ Team Map"],
            label_visibility="collapsed",
        )
        st.divider()
        cov = avg_pos_coverage(season)
        st.metric("Avg-pos coverage", f"{cov['pct']:.0f}%",
                  f"{cov['done']}/{cov['total']}")

    eng = load_engine(season)

    if page.endswith("Overview"):
        page_overview(eng, season)
    elif page.endswith("Similar Players"):
        page_similar(eng)
    elif page.endswith("Squad Completion"):
        page_squad(eng)
    else:
        page_team_map(eng)


if __name__ == "__main__":
    main()
