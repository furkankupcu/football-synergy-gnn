"""Transfer analysis — squad gaps, player similarity, value/production ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


POSITION_GROUPS = {
    "GK": ["GK"],
    "CB": ["CB", "DC"],
    "FB": ["LB", "RB", "LWB", "RWB", "DR", "DL", "WB"],
    "DM": ["DM", "DMF"],
    "CM": ["CM", "MF", "MC"],
    "AM": ["AM", "AMC", "CAM"],
    "W": ["LW", "RW", "AML", "AMR", "WF"],
    "ST": ["ST", "CF", "FW"],
}

PLAYER_FEATURES = [
    "goals", "assists", "xg", "xa", "npxg",
    "progressive_passes", "progressive_carries",
    "tackles", "interceptions", "blocks",
    "shots", "shots_on_target",
    "passes_completed", "passes_pct",
    "carries", "touches",
    "take_ons_won", "aerials_won",
]


def normalize_position(pos) -> str:
    """Map a raw position string to a standard group."""
    if not isinstance(pos, str):
        return "Unknown"
    pos_upper = pos.upper().strip()
    # FBref sometimes gives multiple positions such as 'MF,FW' → take the first
    primary = pos_upper.split(",")[0].split("/")[0].strip()
    for group, variants in POSITION_GROUPS.items():
        if primary in variants:
            return group
        if any(v in primary for v in variants):
            return group
    return "Unknown"


def squad_position_gaps(team_players: pd.DataFrame) -> pd.DataFrame:
    """Analyse the position distribution of a squad and return the gaps."""
    min_required = {"GK": 2, "CB": 3, "FB": 2, "DM": 2, "CM": 2, "AM": 1, "W": 2, "ST": 2}
    df = team_players.copy()
    if "position" not in df.columns:
        return pd.DataFrame()

    df["pos_group"] = df["position"].apply(normalize_position)
    counts = df.groupby("pos_group")["player"].count()

    rows = []
    for pos, req in min_required.items():
        current = int(counts.get(pos, 0))
        rows.append({
            "position": pos,
            "current": current,
            "required": req,
            "gap": max(0, req - current),
            "status": "MISSING" if current < req else "OK",
        })
    return pd.DataFrame(rows)


def build_player_feature_matrix(player_stats: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Player feature matrix (normalised per 90)."""
    available = [f for f in PLAYER_FEATURES if f in player_stats.columns]
    X = player_stats[available].copy().apply(pd.to_numeric, errors="coerce").fillna(0)

    if "minutes" in player_stats.columns:
        minutes = pd.to_numeric(player_stats["minutes"], errors="coerce").replace(0, np.nan)
        scale = 90.0 / minutes
        for col in available:
            X[col] = X[col] * scale
        X = X.fillna(0)
    return X, available


def find_similar_players(
    player_stats: pd.DataFrame,
    target_player: str,
    n: int = 8,
    same_position: bool = True,
    min_minutes: int = 450,
) -> pd.DataFrame:
    """Return the n players most similar to the target player by cosine similarity."""
    df = player_stats.copy()
    if "minutes" in df.columns:
        df = df[pd.to_numeric(df["minutes"], errors="coerce").fillna(0) >= min_minutes]

    if "player" not in df.columns or target_player not in df["player"].values:
        raise ValueError(f"Player not found or not enough minutes: {target_player}")

    df = df.reset_index(drop=True)

    if same_position and "position" in df.columns:
        target_pos = df.loc[df["player"] == target_player, "position"].iloc[0]
        target_group = normalize_position(target_pos)
        df = df[df["position"].apply(normalize_position) == target_group].reset_index(drop=True)

    X, features = build_player_feature_matrix(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    target_pos = df[df["player"] == target_player].index[0]
    sims = cosine_similarity(X_scaled[target_pos].reshape(1, -1), X_scaled)[0]
    df["similarity"] = sims

    similar = df[df["player"] != target_player].nlargest(n, "similarity")
    show_cols = ["player", "team", "position", "minutes", "goals", "assists", "xg", "xa", "similarity"]
    show_cols = [c for c in show_cols if c in similar.columns]
    return similar[show_cols].reset_index(drop=True)


def value_performance_ratio(
    player_stats: pd.DataFrame,
    min_minutes: int = 450,
    market_value_col: str = "market_value",
) -> pd.DataFrame:
    """Market value / production (xG+xA per 90) ratio — high = cheap and productive."""
    df = player_stats.copy()
    if "minutes" in df.columns:
        df = df[pd.to_numeric(df["minutes"], errors="coerce").fillna(0) >= min_minutes]

    xg = pd.to_numeric(df.get("xg", 0), errors="coerce").fillna(0)
    xa = pd.to_numeric(df.get("xa", 0), errors="coerce").fillna(0)
    minutes = pd.to_numeric(df.get("minutes", np.nan), errors="coerce")
    df["xg_xa_per90"] = (xg + xa) / minutes * 90

    if market_value_col in df.columns:
        mv = pd.to_numeric(df[market_value_col], errors="coerce")
        # Low value + high production = high score
        df["value_score"] = df["xg_xa_per90"] / (mv / 1e6).replace(0, np.nan)
    return df.sort_values("value_score", ascending=False, na_position="last")


# ─── Style fit & squad assessment ─────────────────────────────────────────────

# Formation → slot list: each slot is (slot_id, label, position_code, suggested_role).
# position_code: 11-position taxonomy (consistent with the GNN system).
# suggested_role: POSITION_PROFILES key — used when computing a slot's style fit.
FORMATION_SLOTS: dict[str, list[tuple[str, str, str, str]]] = {
    "4-3-3": [
        ("gk",   "Goalkeeper", "GK",  "Goalkeeper"),
        ("lb",   "Left Back",  "LB",  "Full-Back"),
        ("lcb",  "Left CB",    "LCB", "Ball-Playing CB"),
        ("rcb",  "Right CB",   "RCB", "Combative CB / Stopper"),
        ("rb",   "Right Back", "RB",  "Full-Back"),
        ("dm",   "Holding Mid", "CDM", "Defensive Midfielder (#6)"),
        ("lcm",  "Left CM",    "CM",  "Box-to-Box Midfielder"),
        ("rcm",  "Right CM",   "CM",  "Attacking Midfielder (#8)"),
        ("lw",   "Left Wing",  "LW",  "Classic Winger"),
        ("st",   "Striker",    "ST",  "Poacher"),
        ("rw",   "Right Wing", "RW",  "Inverted Winger"),
    ],
    "4-2-3-1": [
        ("gk",   "Goalkeeper",     "GK",  "Goalkeeper"),
        ("lb",   "Left Back",      "LB",  "Full-Back"),
        ("lcb",  "Left CB",        "LCB", "Ball-Playing CB"),
        ("rcb",  "Right CB",       "RCB", "Combative CB / Stopper"),
        ("rb",   "Right Back",     "RB",  "Full-Back"),
        ("ldm",  "Left DM",        "CDM", "Defensive Midfielder (#6)"),
        ("rdm",  "Right DM",       "CDM", "Box-to-Box Midfielder"),
        ("lam",  "Left AM",        "LW",  "Attacking Midfielder (#8)"),
        ("cam",  "Number 10",      "CAM", "Playmaker (#10)"),
        ("ram",  "Right AM",       "RW",  "Attacking Midfielder (#8)"),
        ("st",   "Striker",        "ST",  "Pressing Forward"),
    ],
    "3-5-2": [
        ("gk",   "Goalkeeper", "GK",  "Goalkeeper"),
        ("lcb",  "Left CB",    "LCB", "Wide CB (in 3)"),
        ("ccb",  "Centre CB",  "RCB", "Libero / Middle CB"),
        ("rcb",  "Right CB",   "RCB", "Wide CB (in 3)"),
        ("lwb",  "Left WB",    "LB",  "Wing-Back"),
        ("rwb",  "Right WB",   "RB",  "Wing-Back"),
        ("dm",   "Holding Mid", "CDM", "Deep-Lying Playmaker"),
        ("lcm",  "Left CM",    "CM",  "Box-to-Box Midfielder"),
        ("rcm",  "Right CM",   "CM",  "Attacking Midfielder (#8)"),
        ("ls",   "Left ST",    "ST",  "Target Man (#9)"),
        ("rs",   "Right ST",   "ST",  "Poacher"),
    ],
    "4-4-2": [
        ("gk",   "Goalkeeper", "GK",  "Goalkeeper"),
        ("lb",   "Left Back",  "LB",  "Full-Back"),
        ("lcb",  "Left CB",    "LCB", "Combative CB / Stopper"),
        ("rcb",  "Right CB",   "RCB", "Combative CB / Stopper"),
        ("rb",   "Right Back", "RB",  "Full-Back"),
        ("lm",   "Left Wing",  "LW",  "Classic Winger"),
        ("lcm",  "Left CM",    "CM",  "Box-to-Box Midfielder"),
        ("rcm",  "Right CM",   "CM",  "Deep-Lying Playmaker"),
        ("rm",   "Right Wing", "RW",  "Classic Winger"),
        ("ls",   "Left ST",    "ST",  "Target Man (#9)"),
        ("rs",   "Right ST",   "ST",  "Poacher"),
    ],
}


def get_formation_slots(formation: str) -> list[dict]:
    """Formation ID → slot list for the API (as dicts)."""
    raw = FORMATION_SLOTS.get(formation, FORMATION_SLOTS["4-3-3"])
    return [
        {"slot_id": sid, "label": lbl, "position_code": pos, "suggested_role": role}
        for (sid, lbl, pos, role) in raw
    ]


def compute_style_fit(
    player_metrics: dict[str, float],
    target_vector: dict[str, float] | "pd.Series",
) -> float:
    """Fit (0-100) between a player's metric-percentile dict and the target vector.

    Computation: mean absolute difference over shared metrics, subtracted from 100.
    """
    if hasattr(target_vector, "to_dict"):
        target_vector = target_vector.to_dict()
    common = [m for m in target_vector if m in player_metrics]
    if not common:
        return 0.0
    diffs = [abs(float(player_metrics[m]) - float(target_vector[m])) for m in common]
    return round(max(0.0, 100.0 - sum(diffs) / len(diffs)), 1)


def evaluate_squad_fit(
    team_players: pd.DataFrame,
    formation: str,
    target_vector: pd.Series | None,
    slot_assignments: dict[str, int | str] | None = None,
) -> list[dict]:
    """Return the style fit of the placed player for every slot + (if empty) the need for a suggestion.

    Args:
        team_players: player pool (current team); minimum columns:
                      source_player_id, player_name, position + STYLE_FEATURES
                      with the same name as percentile columns are NOT expected — the metric
                      columns are expected here; percentiles are computed (league rank).
        formation: "4-3-3" etc.
        target_vector: percentile Series on the STYLE_FEATURES axis. None → no fit.
        slot_assignments: {slot_id: player_id?} — players placed by the user.

    Returns:
        Slot list of dicts: {slot_id, label, position_code, suggested_role,
        assigned_player, fit_pct, reason}.
    """
    slots = get_formation_slots(formation)
    slot_assignments = slot_assignments or {}

    out = []
    for slot in slots:
        sid = slot["slot_id"]
        pid = slot_assignments.get(sid)
        rec = dict(slot)
        rec["assigned_player"] = None
        rec["fit_pct"] = None
        rec["reason"] = None

        if pid is not None and not team_players.empty:
            row = team_players[
                (team_players.get("source_player_id", -1) == pid)
                | (team_players.get("player_name", "") == pid)
            ]
            if not row.empty:
                p = row.iloc[0].to_dict()
                rec["assigned_player"] = {
                    "id": p.get("source_player_id"),
                    "name": p.get("player_name"),
                    "position": p.get("position"),
                    "rating": p.get("rating"),
                }
                if target_vector is not None:
                    # Compute the player's role-metric percentile within the league (approximation).
                    # In practice the target_vector index is TEAM-level STYLE_FEATURES; players
                    # do not have them 1:1. Use a coarse proxy instead:
                    #   player_metrics[m] = p[m] if present, else 50.
                    p_metrics = {m: float(p[m]) if m in p and pd.notna(p[m]) else 50.0
                                 for m in target_vector.index}
                    rec["fit_pct"] = compute_style_fit(p_metrics, target_vector)
                    if rec["fit_pct"] >= 70:
                        rec["reason"] = "Strong fit with the style."
                    elif rec["fit_pct"] >= 50:
                        rec["reason"] = "Borderline fit — short on some metrics."
                    else:
                        rec["reason"] = "Does not fit the style — rotation/sale candidate."
        out.append(rec)
    return out


def radar_data(player_stats: pd.DataFrame, players: list[str]) -> pd.DataFrame:
    """Return league-percentile DataFrame for the selected players (for radar)."""
    X, features = build_player_feature_matrix(player_stats)
    percentiles = (X.rank(pct=True) * 100).round(1)
    mask = player_stats["player"].isin(players)
    result = percentiles[mask.values].copy()
    result.index = player_stats.loc[mask, "player"].values
    return result[features]
