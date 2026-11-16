"""Position profiles and role-score computation — single source.

This module used to be duplicated in `src/app/dashboard.py` (Streamlit, removed) and
`src/api/main.py`. Both sides now import it from here.

"""

from __future__ import annotations

import numpy as np
import pandas as pd


POSITION_PROFILES: dict = {
    "Ball-Playing CB": {
        "group": "Center Backs",
        "metrics": {
            "pass_p90":        ("Passes p90",             "total_pass",                       "volume"),
            "prog_pass_p90":   ("Prog. Passes p90",       "total_progression",                "volume"),
            "prog_pass_pct":   ("Prog. Pass Acc. %",      "_prog_pass_pct",                   "pct"),
            "long_acc_pct":    ("Long Pass Acc. %",       "_long_acc_pct",                    "pct"),
            "interc_p90":      ("Interceptions p90",      "interception_won",                 "volume"),
        },
    },
    "Combative CB / Stopper": {
        "group": "Center Backs",
        "metrics": {
            "def_duel_p90":    ("Def. Duels p90",         "_def_duels",                       "volume"),
            "def_duel_pct":    ("Def. Duels Won %",       "_def_duel_pct",                    "pct"),
            "aerial_p90":      ("Aerial Duels p90",       "_aerials",                         "volume"),
            "aerial_pct":      ("Aerial Won %",           "_aerial_pct",                      "pct"),
            "block_p90":       ("Shots Blocked p90",      "outfielder_block",                 "volume"),
        },
    },
    "Libero / Middle CB": {
        "group": "Center Backs",
        "metrics": {
            "def_duel_p90":    ("Def. Duels p90",         "_def_duels",                       "volume"),
            "def_duel_pct":    ("Def. Duels Won %",       "_def_duel_pct",                    "pct"),
            "interc_p90":      ("Interceptions p90",      "interception_won",                 "volume"),
            "pass_acc_pct":    ("Pass Acc. %",            "_pass_acc_pct",                    "pct"),
            "long_acc_pct":    ("Long Pass Acc. %",       "_long_acc_pct",                    "pct"),
        },
    },
    "Wide CB (in 3)": {
        "group": "Center Backs",
        "metrics": {
            "def_duel_p90":    ("Def. Duels p90",         "_def_duels",                       "volume"),
            "def_duel_pct":    ("Def. Duels Won %",       "_def_duel_pct",                    "pct"),
            "interc_p90":      ("Interceptions p90",      "interception_won",                 "volume"),
            "prog_pass_p90":   ("Prog. Passes p90",       "total_progression",                "volume"),
            "prog_pass_pct":   ("Prog. Pass Acc. %",      "_prog_pass_pct",                   "pct"),
        },
    },
    "Defensive Midfielder (#6)": {
        "group": "Midfielders",
        "metrics": {
            "interc_p90":      ("Interceptions p90",      "interception_won",                 "volume"),
            "def_duel_p90":    ("Def. Duels p90",         "_def_duels",                       "volume"),
            "def_duel_pct":    ("Def. Duels Won %",       "_def_duel_pct",                    "pct"),
            "pass_acc_pct":    ("Pass Acc. %",            "_pass_acc_pct",                    "pct"),
        },
    },
    "Attacking Midfielder (#8)": {
        "group": "Midfielders",
        "metrics": {
            "prog_pass_p90":   ("Prog. Passes p90",       "total_progression",                "volume"),
            "prog_pass_pct":   ("Prog. Pass Acc. %",      "_prog_pass_pct",                   "pct"),
            "prog_run_p90":    ("Prog. Runs p90",         "progressive_ball_carries_count",   "volume"),
            "xa_p90":          ("xA p90",                 "expected_assists",                 "volume"),
            "shot_p90":        ("Shots p90",              "total_shots",                      "volume"),
        },
    },
    "Deep-Lying Playmaker": {
        "group": "Midfielders",
        "metrics": {
            "prog_pass_p90":   ("Prog. Passes p90",       "total_progression",                "volume"),
            "prog_pass_pct":   ("Prog. Pass Acc. %",      "_prog_pass_pct",                   "pct"),
            "long_acc_pct":    ("Long Pass Acc. %",       "_long_acc_pct",                    "pct"),
            "interc_p90":      ("Interceptions p90",      "interception_won",                 "volume"),
        },
    },
    "Box-to-Box Midfielder": {
        "group": "Midfielders",
        "metrics": {
            "def_duel_p90":    ("Def. Duels p90",         "_def_duels",                       "volume"),
            "def_duel_pct":    ("Def. Duels Won %",       "_def_duel_pct",                    "pct"),
            "prog_run_p90":    ("Prog. Runs p90",         "progressive_ball_carries_count",   "volume"),
            "shot_p90":        ("Shots p90",              "total_shots",                      "volume"),
            "xa_p90":          ("xA p90",                 "expected_assists",                 "volume"),
        },
    },
    "Full-Back": {
        "group": "Wide Players",
        "metrics": {
            "def_duel_p90":    ("Def. Duels p90",         "_def_duels",                       "volume"),
            "def_duel_pct":    ("Def. Duels Won %",       "_def_duel_pct",                    "pct"),
            "interc_p90":      ("Interceptions p90",      "interception_won",                 "volume"),
            "cross_p90":       ("Crosses p90",            "total_cross",                      "volume"),
            "cross_acc_pct":   ("Cross Acc. %",           "_cross_acc_pct",                   "pct"),
        },
    },
    "Wing-Back": {
        "group": "Wide Players",
        "metrics": {
            "interc_p90":      ("Interceptions p90",      "interception_won",                 "volume"),
            "prog_run_p90":    ("Prog. Runs p90",         "progressive_ball_carries_count",   "volume"),
            "cross_p90":       ("Crosses p90",            "total_cross",                      "volume"),
            "cross_acc_pct":   ("Cross Acc. %",           "_cross_acc_pct",                   "pct"),
            "key_pass_p90":    ("Shot Assists p90",       "key_pass",                         "volume"),
        },
    },
    "Classic Winger": {
        "group": "Wide Players",
        "metrics": {
            "dribble_p90":     ("Dribbles p90",           "_dribbles",                        "volume"),
            "dribble_pct":     ("Dribble Succ. %",        "_dribble_pct",                     "pct"),
            "prog_run_p90":    ("Prog. Runs p90",         "progressive_ball_carries_count",   "volume"),
            "cross_p90":       ("Crosses p90",            "total_cross",                      "volume"),
            "key_pass_p90":    ("Shot Assists p90",       "key_pass",                         "volume"),
        },
    },
    "Inverted Winger": {
        "group": "Wide Players",
        "metrics": {
            "shot_p90":        ("Shots p90",              "total_shots",                      "volume"),
            "xg_p90":          ("xG p90",                 "expected_goals",                   "volume"),
            "prog_run_p90":    ("Prog. Runs p90",         "progressive_ball_carries_count",   "volume"),
            "key_pass_p90":    ("Shot Assists p90",       "key_pass",                         "volume"),
            "xa_p90":          ("xA p90",                 "expected_assists",                 "volume"),
        },
    },
    "Playmaker (#10)": {
        "group": "Forwards",
        "metrics": {
            "prog_pass_p90":   ("Prog. Passes p90",       "total_progression",                "volume"),
            "prog_pass_pct":   ("Prog. Pass Acc. %",      "_prog_pass_pct",                   "pct"),
            "opp_half_p90":    ("Deep Completions p90",   "accurate_opposition_half_passes",  "volume"),
            "key_pass_p90":    ("Shot Assists p90",       "key_pass",                         "volume"),
            "xa_p90":          ("xA p90",                 "expected_assists",                 "volume"),
            "shot_p90":        ("Shots p90",              "total_shots",                      "volume"),
        },
    },
    "Target Man (#9)": {
        "group": "Forwards",
        "metrics": {
            "aerial_pct":      ("Aerial Won %",           "_aerial_pct",                      "pct"),
            "aerial_p90":      ("Aerial Duels p90",       "_aerials",                         "volume"),
            "fouled_p90":      ("Fouls Suffered p90",     "was_fouled",                       "volume"),
            "xg_p90":          ("xG p90",                 "expected_goals",                   "volume"),
        },
    },
    "Poacher": {
        "group": "Forwards",
        "metrics": {
            "xg_p90":          ("xG p90",                 "expected_goals",                   "volume"),
            "goal_p90":        ("Goals p90",              "goals",                            "volume"),
            "conv_pct":        ("Goal Conversion %",      "_conv_pct",                        "pct"),
        },
    },
    "Pressing Forward": {
        "group": "Forwards",
        "metrics": {
            "def_duel_p90":    ("Def. Duels p90",         "_def_duels",                       "volume"),
            "interc_p90":      ("Interceptions p90",      "interception_won",                 "volume"),
            "shot_p90":        ("Shots p90",              "total_shots",                      "volume"),
            "prog_run_p90":    ("Prog. Runs p90",         "progressive_ball_carries_count",   "volume"),
            "xa_p90":          ("xA p90",                 "expected_assists",                 "volume"),
        },
    },
    "Goalkeeper": {
        "group": "Goalkeepers",
        "metrics": {
            "aerial_pct":      ("Aerial Won %",           "_aerial_pct",                      "pct"),
            "aerial_p90":      ("Aerial Duels p90",       "_aerials",                         "volume"),
            "interc_p90":      ("Interceptions p90",      "interception_won",                 "volume"),
            "block_p90":       ("Blocks p90",             "outfielder_block",                 "volume"),
            "duel_p90":        ("Def. Duels p90",         "_def_duels",                       "volume"),
        },
    },
}

PROFILE_GROUPS: dict[str, list[str]] = {
    "Goalkeepers":  ["Goalkeeper"],
    "Center Backs": ["Ball-Playing CB", "Combative CB / Stopper", "Libero / Middle CB", "Wide CB (in 3)"],
    "Midfielders":  ["Defensive Midfielder (#6)", "Attacking Midfielder (#8)", "Deep-Lying Playmaker", "Box-to-Box Midfielder"],
    "Wide Players": ["Full-Back", "Wing-Back", "Classic Winger", "Inverted Winger"],
    "Forwards":     ["Playmaker (#10)", "Target Man (#9)", "Poacher", "Pressing Forward"],
}

# Position code (G/D/M/F) → scouting roles.
# Identical to _POS_ROLES in api/main.py; the single source lives here.
POS_ROLES: dict[str, list[str]] = {
    "G": ["Goalkeeper"],
    "D": ["Ball-Playing CB", "Combative CB / Stopper", "Full-Back", "Wing-Back"],
    "M": ["Defensive Midfielder (#6)", "Attacking Midfielder (#8)", "Deep-Lying Playmaker", "Box-to-Box Midfielder"],
    "F": ["Playmaker (#10)", "Target Man (#9)", "Poacher", "Pressing Forward"],
}

POS_REQUIRED: dict[str, int] = {"G": 2, "D": 4, "M": 4, "F": 2}


# 11 detailed positions (GNN coordinate-based taxonomy) → scouting roles.
# Consistent with src/graph/positions.py. The web side should use this structure.
POS_ROLES_DETAILED: dict[str, list[str]] = {
    "GK":  ["Goalkeeper"],
    "RB":  ["Full-Back", "Wing-Back"],
    "LB":  ["Full-Back", "Wing-Back"],
    "RCB": ["Ball-Playing CB", "Combative CB / Stopper", "Libero / Middle CB", "Wide CB (in 3)"],
    "LCB": ["Ball-Playing CB", "Combative CB / Stopper", "Libero / Middle CB", "Wide CB (in 3)"],
    "CDM": ["Defensive Midfielder (#6)", "Deep-Lying Playmaker"],
    "CM":  ["Box-to-Box Midfielder", "Deep-Lying Playmaker", "Defensive Midfielder (#6)"],
    "CAM": ["Attacking Midfielder (#8)", "Playmaker (#10)"],
    "RW":  ["Classic Winger", "Inverted Winger"],
    "LW":  ["Classic Winger", "Inverted Winger"],
    "ST":  ["Target Man (#9)", "Poacher", "Pressing Forward"],
}

POS_REQUIRED_DETAILED: dict[str, int] = {
    "GK": 2, "RB": 2, "LB": 2, "RCB": 2, "LCB": 2,
    "CDM": 1, "CM": 2, "CAM": 1, "RW": 2, "LW": 2, "ST": 2,
}

# Default mapping from the old G/D/M/F code to the 11 positions
# (fallback when there is no avg_pos data)
POS_BASE_TO_DETAILED: dict[str, str] = {
    "G": "GK", "D": "RCB", "M": "CM", "F": "ST",
}


def compute_role_scores(df: pd.DataFrame, met_defs: dict) -> pd.Series:
    """0-100 role score from the metrics of the position profile.

    Formula: role_pct * 0.80 + rating_pct * 0.20, weighted by log(matches_played).
    """
    if df.empty:
        return pd.Series(dtype=float)

    metric_cols = [col for _, col, _ in met_defs.values() if col in df.columns]
    if not metric_cols:
        return pd.Series(0.0, index=df.index)

    pct_parts = [df[col].fillna(0).rank(pct=True) * 100 for col in metric_cols]
    rol_pct = pd.concat(pct_parts, axis=1).mean(axis=1)

    if "rating" in df.columns:
        rating_pct = df["rating"].fillna(0).rank(pct=True) * 100
    else:
        rating_pct = pd.Series(50.0, index=df.index)

    ham = rol_pct * 0.80 + rating_pct * 0.20

    if "matches_played" in df.columns:
        mp = df["matches_played"].fillna(0).clip(lower=0)
        max_mp = mp.max()
        log_w = (np.log1p(mp) / np.log1p(max_mp)) if max_mp > 0 else pd.Series(1.0, index=df.index)
    else:
        log_w = pd.Series(1.0, index=df.index)

    return (ham * log_w).round(2)
