"""PyG HeteroData graph builder.

Data sources:
- Node features: `db.get_player_stats(season=..)` → per-90 stats + physical + position
- Edges: `data/graph/edges_teammate_<season>.parquet` (phase 2 output)

Output: `data/graph/hetero_<season>.pt` (torch.save) + node_id↔player_id map (.json)
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src import db
from src.graph.edge_builder import _season_where
from src.graph.positions import (
    coordinate_to_position,
    normalize_position,
    position_one_hot,
)

GRAPH_DIR = Path("data/graph")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Node feature columns (normalised to per-90, missing values imputed)
NUMERIC_FEATURES = [
    "goals", "assists", "xg", "xa", "npxg",
    "progressive_carries", "progressive_passes",
    "shots", "shots_on_target",
    "tackles", "tackles_won", "interceptions", "blocks", "clearances",
    "passes_completed", "passes_pct", "assisted_shots",
    "passes_into_final_third", "passes_into_penalty_area",
    "touches", "carries", "take_ons_attempted", "take_ons_won",
    "aerials_won", "aerials_lost",
    "fouls", "fouled", "cards_yellow", "cards_red",
]
PHYSICAL_FEATURES = ["age", "height"]
FOOT_VALUES = ["left", "right", "both"]


def _per90(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise stats to per-90 using minutes."""
    out = df.copy()
    minutes = out["minutes"].replace(0, np.nan)
    for c in NUMERIC_FEATURES:
        if c in out.columns and c != "passes_pct":  # passes_pct zaten oran
            out[c] = out[c] / minutes * 90.0
    return out


def _impute(df: pd.DataFrame, group_cols: list[str], cols: list[str]) -> pd.DataFrame:
    """Impute with the league+position median; remaining NaNs become 0."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
            continue
        out[c] = out.groupby(group_cols)[c].transform(lambda s: s.fillna(s.median()))
        out[c] = out[c].fillna(out[c].median()).fillna(0.0)
    return out


SHRINK_MINUTES = 450  # prior strength ≈ five full matches


def _shrink(df: pd.DataFrame, group_cols: list[str], cols: list[str],
            k: float = SHRINK_MINUTES) -> pd.DataFrame:
    """Pull per-90 rates of low-minute players towards their league × position average.

    x' = (minutes · x + k · prior) / (minutes + k), prior = minutes-weighted group mean.
    Without it a 20-minute cameo with one goal reads as 4.5 goals per 90, and the model
    extrapolates such players into implausibly high synergy with everyone.
    """
    out = df.copy()
    m = out["minutes"].fillna(0).clip(lower=0)
    for c in cols:
        w = out[c] * m
        prior = (w.groupby([out[g] for g in group_cols]).transform("sum")
                 / m.groupby([out[g] for g in group_cols]).transform("sum").replace(0, np.nan))
        prior = prior.fillna((w.sum() / m.sum()) if m.sum() else 0.0)
        out[c] = (m * out[c] + k * prior) / (m + k)
    return out


def _foot_one_hot(series: pd.Series) -> np.ndarray:
    s = series.fillna("").astype(str).str.lower()
    return np.stack([(s == v).astype(np.float32).values for v in FOOT_VALUES], axis=1)


def _build_node_features(season: str, league: str | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    raw = db.get_player_stats_from_source(league=league, season=season)
    if raw.empty:
        raise RuntimeError(f"{season}/{league}: no player stats found")

    # In-season transfers: assign the player to the **last team he played for** that season.
    # NOTE: this query must be limited to the same season — otherwise a team from another
    # season does not match and a transferred player is dropped from the graph entirely.
    where_sql, params = _season_where(season, league)
    extra = "AND" if where_sql else "WHERE"
    last_team = db.read_sql(f"""
        SELECT 'ss_' || p.source_player_id::text AS player_id,
               p.team_name                          AS team,
               MAX(m.match_date)                    AS last_match
        FROM player_stats p
        JOIN matches m ON m.match_id = p.match_id
        {where_sql}
        {extra} p.minutes_played > 0
        GROUP BY p.source_player_id, p.team_name
    """, params)
    last_team = (last_team.sort_values("last_match", ascending=False)
                          .drop_duplicates("player_id", keep="first")
                          [["player_id", "team"]]
                          .rename(columns={"team": "_last_team"}))
    raw = raw.merge(last_team, on="player_id", how="left")
    raw = raw[raw["team"] == raw["_last_team"]].drop(columns=["_last_team"])
    raw = raw.drop_duplicates("player_id", keep="first")

    df = _per90(raw)
    df = _impute(df, group_cols=["league", "position"], cols=NUMERIC_FEATURES + PHYSICAL_FEATURES)
    df = _shrink(df, group_cols=["league", "position"],
                 cols=[c for c in NUMERIC_FEATURES if c in df.columns])

    # Position — coordinate-based detailed position when avg_pos data exists,
    # otherwise base_code (G/D/M/F) → normalised fallback
    avg_pos = db.read_sql(f"""
        SELECT
            'ss_' || a.source_player_id::text AS player_id,
            SUM(a.avg_x * COALESCE(a.points_count, 1)) / NULLIF(SUM(COALESCE(a.points_count, 1)), 0) AS avg_x,
            SUM(a.avg_y * COALESCE(a.points_count, 1)) / NULLIF(SUM(COALESCE(a.points_count, 1)), 0) AS avg_y
        FROM player_avg_positions a
        JOIN matches m ON m.match_id = a.match_id
        {where_sql}
        GROUP BY a.source_player_id
    """, params)
    df = df.merge(avg_pos, on="player_id", how="left")

    def _pos(r):
        if pd.notna(r.get("avg_x")) and pd.notna(r.get("avg_y")):
            return coordinate_to_position(r["avg_x"], r["avg_y"], r.get("position"))
        return normalize_position(r.get("tm_position"), r.get("position"))

    df["pos_norm"] = df.apply(_pos, axis=1)
    pos_oh = np.stack([position_one_hot(p) for p in df["pos_norm"]], axis=0).astype(np.float32)

    foot_oh = _foot_one_hot(df["foot"]) if "foot" in df.columns else np.zeros((len(df), len(FOOT_VALUES)), dtype=np.float32)

    num = df[NUMERIC_FEATURES + PHYSICAL_FEATURES].to_numpy(dtype=np.float32)

    # Z-score normalise numeric features (position one-hot stays as is)
    mu = num.mean(axis=0, keepdims=True)
    sd = num.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    num_norm = (num - mu) / sd

    x = np.concatenate([num_norm, foot_oh, pos_oh], axis=1).astype(np.float32)
    return df.reset_index(drop=True), x


def build_hetero_graph(season: str, league: str | None = None,
                       target: str = "uplift") -> HeteroData:
    """target: 'uplift' (recommended, true synergy) | 'joint' (legacy, star-biased)."""
    nodes_df, x = _build_node_features(season, league)
    pid_to_idx = {pid: i for i, pid in enumerate(nodes_df["player_id"])}

    data = HeteroData()
    data["player"].x = torch.from_numpy(x)
    data["player"].player_id = nodes_df["player_id"].tolist()
    name_col = "player_name" if "player_name" in nodes_df else ("player" if "player" in nodes_df else None)
    data["player"].player_name = nodes_df[name_col].tolist() if name_col else nodes_df["player_id"].tolist()
    data["player"].position = nodes_df["pos_norm"].tolist()
    data["player"].team = nodes_df["team"].tolist() if "team" in nodes_df else ["?"] * len(nodes_df)

    # Phase 1.4: categorical team + league ids (for learned embeddings)
    teams_list = data["player"].team
    leagues_list = nodes_df["league"].tolist() if "league" in nodes_df else ["?"] * len(nodes_df)
    team_vocab = {t: i for i, t in enumerate(sorted(set(teams_list)))}
    league_vocab = {l: i for i, l in enumerate(sorted(set(leagues_list)))}
    data["player"].team_idx = torch.tensor([team_vocab[t] for t in teams_list], dtype=torch.long)
    data["player"].league_idx = torch.tensor([league_vocab[l] for l in leagues_list], dtype=torch.long)
    data["player"].num_teams = len(team_vocab)
    data["player"].num_leagues = len(league_vocab)
    print(f"[graph] teams={len(team_vocab)} leagues={len(league_vocab)}")

    # Edges
    teammate_path = GRAPH_DIR / f"edges_teammate_{season}.parquet"
    if teammate_path.exists():
        ed = pd.read_parquet(teammate_path)
        ed = ed[ed["player_a"].isin(pid_to_idx) & ed["player_b"].isin(pid_to_idx)]
        src = ed["player_a"].map(pid_to_idx).to_numpy()
        dst = ed["player_b"].map(pid_to_idx).to_numpy()
        # Undirected: add both directions
        edge_index = torch.tensor(
            np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0),
            dtype=torch.long,
        )
        attr_cols = ["minutes_together", "matches_together", "joint_g", "joint_a",
                     "joint_xg", "joint_xa", "per90_joint_output"]
        attr_vals = ed[attr_cols].to_numpy(dtype=np.float32)
        attr_vals = np.concatenate([attr_vals, attr_vals], axis=0)  # for both directions

        label_col = "per90_uplift" if target == "uplift" else "per90_joint_output"
        if label_col not in ed.columns:
            print(f"[graph] '{label_col}' missing, falling back to 'per90_joint_output'")
            label_col = "per90_joint_output"
        labels = ed[label_col].to_numpy(dtype=np.float32)
        labels = np.concatenate([labels, labels], axis=0)
        print(f"[graph] target={target} | label_col={label_col} | "
              f"y_stats: mean={labels.mean():.3f} std={labels.std():.3f} "
              f"min={labels.min():.3f} max={labels.max():.3f}")

        data["player", "teammate", "player"].edge_index = edge_index
        data["player", "teammate", "player"].edge_attr = torch.from_numpy(attr_vals)
        data["player", "teammate", "player"].y = torch.from_numpy(labels)
        if "def_uplift" in ed.columns:  # defensive target (xG conceded together); NaN = no shot data
            y_def = ed["def_uplift"].to_numpy(dtype=np.float32)
            data["player", "teammate", "player"].y_def = torch.from_numpy(np.concatenate([y_def, y_def]))
    else:
        print(f"[graph] teammate edges parquet missing: {teammate_path}")

    opp_path = GRAPH_DIR / f"edges_opponent_{season}.parquet"
    if opp_path.exists():
        ed = pd.read_parquet(opp_path)
        ed = ed[ed["player_a"].isin(pid_to_idx) & ed["player_b"].isin(pid_to_idx)]
        src = ed["player_a"].map(pid_to_idx).to_numpy()
        dst = ed["player_b"].map(pid_to_idx).to_numpy()
        edge_index = torch.tensor(
            np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0),
            dtype=torch.long,
        )
        data["player", "opponent", "player"].edge_index = edge_index

    # Persist
    out_pt = GRAPH_DIR / f"hetero_{season}.pt"
    torch.save(data, out_pt)
    (GRAPH_DIR / f"node_index_{season}.json").write_text(json.dumps(pid_to_idx, indent=2))
    print(f"[graph] saved {out_pt} | nodes={len(nodes_df)} feature_dim={x.shape[1]}")
    return data


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--season", default="2024-2025")
    p.add_argument("--league", default=None)
    p.add_argument("--target", choices=["uplift", "joint"], default="uplift",
                   help="uplift: true synergy signal | joint: legacy, star-biased")
    args = p.parse_args()
    build_hetero_graph(args.season, args.league, target=args.target)
