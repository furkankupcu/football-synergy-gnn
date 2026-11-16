"""Predefined playing-style templates.

Each template holds target percentiles (0-100) for the metrics in
`src.analysis.style.STYLE_FEATURES`. A team's fit to a template is measured
by cosine similarity with its `style_fingerprint` output.

Template usage:
    >>> from src.analysis.style_templates import STYLE_TEMPLATES, get_template_vector
    >>> target = get_template_vector("high-press")
    >>> # target: pd.Series indeksli (feature_name → 0-100 percentile)
"""

from __future__ import annotations

import pandas as pd

from src.analysis.style import STYLE_FEATURES


# Each template: target percentile (0-100) for the columns in STYLE_FEATURES.
# High value = "we want to be in the league's top percentiles for this metric".
# Missing metrics count as 50 (neutral).
STYLE_TEMPLATES: dict[str, dict] = {
    "high-press": {
        "label": "High Press / Gegenpressing",
        "description": "Immediate ball recovery after losing it, high defensive activity, short-passing control.",
        "targets": {
            "tackles": 85, "interceptions": 85, "blocks": 75,
            "take_ons_won": 70, "take_ons_attempted": 75,
            "passes_pct": 75, "possession": 70,
            "progressive_passes": 70, "progressive_carries": 70,
            "passes_into_final_third": 75, "passes_into_penalty_area": 70,
            "shots": 70, "shots_on_target": 70,
            "xg": 75, "xg_against": 25,
            "goals_for": 75, "goals_against": 25,
            "average_shot_distance": 30,  # close-range shots
        },
    },
    "possession": {
        "label": "Positional Play (Tiki-Taka)",
        "description": "High possession, short and accurate passing, patient territorial gains.",
        "targets": {
            "possession": 90, "passes_pct": 90,
            "progressive_passes": 80, "progressive_carries": 70,
            "passes_into_final_third": 80, "passes_into_penalty_area": 75,
            "tackles": 40, "interceptions": 45, "blocks": 40,
            "take_ons_attempted": 60, "take_ons_won": 65,
            "shots": 65, "shots_on_target": 65,
            "xg": 70, "xg_against": 30,
            "goals_for": 70, "goals_against": 35,
            "average_shot_distance": 40,
        },
    },
    "counter-attack": {
        "label": "Direct / Counter-Attack",
        "description": "Low block, fast transitions, vertical passes, attacks with few touches.",
        "targets": {
            "possession": 30, "passes_pct": 60,
            "progressive_passes": 75, "progressive_carries": 80,
            "passes_into_final_third": 65, "passes_into_penalty_area": 65,
            "tackles": 60, "interceptions": 75, "blocks": 65,
            "take_ons_attempted": 50, "take_ons_won": 60,
            "shots": 50, "shots_on_target": 60,
            "xg": 60, "xg_against": 45,
            "goals_for": 60, "goals_against": 45,
            "average_shot_distance": 35,
        },
    },
    "defensive-block": {
        "label": "Defensive Block",
        "description": "Compact defence, low possession, built around conceding few goals.",
        "targets": {
            "possession": 25, "passes_pct": 65,
            "progressive_passes": 35, "progressive_carries": 35,
            "passes_into_final_third": 35, "passes_into_penalty_area": 30,
            "tackles": 80, "interceptions": 85, "blocks": 85,
            "take_ons_attempted": 30, "take_ons_won": 35,
            "shots": 35, "shots_on_target": 40,
            "xg": 35, "xg_against": 20,
            "goals_for": 40, "goals_against": 20,
            "average_shot_distance": 60,
        },
    },
    "open-game": {
        "label": "Open Game / High Scoring",
        "description": "Open at both ends, many shots/xG but defensively fragile.",
        "targets": {
            "possession": 55, "passes_pct": 70,
            "progressive_passes": 75, "progressive_carries": 80,
            "passes_into_final_third": 80, "passes_into_penalty_area": 80,
            "tackles": 50, "interceptions": 50, "blocks": 50,
            "take_ons_attempted": 70, "take_ons_won": 65,
            "shots": 85, "shots_on_target": 80,
            "xg": 85, "xg_against": 70,
            "goals_for": 85, "goals_against": 65,
            "average_shot_distance": 45,
        },
    },
}


def list_templates() -> list[dict]:
    """For the API: list of ID + label + description."""
    return [
        {"id": tid, "label": t["label"], "description": t["description"]}
        for tid, t in STYLE_TEMPLATES.items()
    ]


def get_template_vector(template_id: str) -> pd.Series:
    """Template ID → Series of target percentiles in STYLE_FEATURES order.

    Missing metrics are filled with 50 (neutral).
    """
    if template_id not in STYLE_TEMPLATES:
        raise ValueError(f"Unknown template: {template_id}")
    tgt = STYLE_TEMPLATES[template_id]["targets"]
    return pd.Series({f: float(tgt.get(f, 50)) for f in STYLE_FEATURES})


def parse_style_target(style_target: str | None, team_stats: pd.DataFrame | None = None) -> pd.Series | None:
    """General parser: resolves the 'template:high-press' or 'team:PSG' formats.

    Returning None means no style target was selected.
    """
    if not style_target:
        return None
    if style_target.startswith("template:"):
        return get_template_vector(style_target.split(":", 1)[1])
    if style_target.startswith("team:") and team_stats is not None:
        from src.analysis.style import style_fingerprint
        team_name = style_target.split(":", 1)[1]
        fp = style_fingerprint(team_stats, team_name)
        if fp.empty:
            return None
        return fp.reindex(STYLE_FEATURES).fillna(50)
    return None
