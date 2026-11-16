"""Automatic team / head-to-head tactical text generator — rule-based.

Feeds the "Why successful?" panel and the H2H "how to win / what to watch out for"
output. Computation: highest/lowest metrics of the style fingerprint (percentile)
and the asymmetry between two teams.
"""

from __future__ import annotations

import pandas as pd


# Metric naming (display label + direction of interpretation)
# direction: "high_good" → a high value is positive
#            "low_good"  → a low value is positive (e.g. xg_against, goals_against, fouls)
METRIC_LABELS: dict[str, tuple[str, str]] = {
    "goals_for":               ("Goals scored",              "high_good"),
    "goals_against":           ("Goals conceded",            "low_good"),
    "xg":                      ("Expected goals (xG)",       "high_good"),
    "xg_against":              ("xG conceded",               "low_good"),
    "possession":              ("Possession",                "high_good"),
    "passes_pct":              ("Pass accuracy",             "high_good"),
    "shots":                   ("Shots",                     "high_good"),
    "shots_on_target":         ("Shots on target",           "high_good"),
    "progressive_passes":      ("Progressive passes",        "high_good"),
    "progressive_carries":     ("Progressive carries",       "high_good"),
    "passes_into_final_third": ("Passes into final third",   "high_good"),
    "passes_into_penalty_area":("Passes into the box",       "high_good"),
    "tackles":                 ("Tackles",                   "high_good"),
    "interceptions":           ("Interceptions",             "high_good"),
    "blocks":                  ("Blocks",                    "high_good"),
    "take_ons_attempted":      ("Dribbles attempted",        "high_good"),
    "take_ons_won":            ("Successful dribbles",       "high_good"),
    "average_shot_distance":   ("Shot distance",             "low_good"),
}


def team_strengths_weaknesses(
    fingerprint: pd.Series,
    high_threshold: float = 70.0,
    low_threshold: float = 30.0,
) -> dict:
    """Split a style fingerprint into strengths and weaknesses.

    Args:
        fingerprint: pd.Series — feature_name → 0-100 league percentile.

    Returns:
        {"strengths": [(metric, label, pct, sentence)],
         "weaknesses": [...],
         "summary": "short summary text"}
    """
    if fingerprint.empty:
        return {"strengths": [], "weaknesses": [], "summary": "Veri yetersiz."}

    strengths, weaknesses = [], []
    for metric, pct in fingerprint.items():
        label_info = METRIC_LABELS.get(metric)
        if not label_info:
            continue
        label, direction = label_info
        effective_pct = pct if direction == "high_good" else (100 - pct)

        if effective_pct >= high_threshold:
            sentence = f"{label} in the league's top {int(100 - effective_pct + (effective_pct))}% — a strength."
            sentence = f"{label}: {int(pct)}th percentile in the league."
            strengths.append({"metric": metric, "label": label, "percentile": float(pct), "sentence": sentence})
        elif effective_pct <= low_threshold:
            sentence = f"{label}: {int(pct)}th percentile in the league — a weakness."
            weaknesses.append({"metric": metric, "label": label, "percentile": float(pct), "sentence": sentence})

    strengths.sort(key=lambda x: -x["percentile"] if METRIC_LABELS[x["metric"]][1] == "high_good" else x["percentile"])
    weaknesses.sort(key=lambda x: x["percentile"] if METRIC_LABELS[x["metric"]][1] == "high_good" else -x["percentile"])

    summary = _build_summary_sentence(strengths[:3], weaknesses[:3])
    return {"strengths": strengths[:6], "weaknesses": weaknesses[:6], "summary": summary}


def _build_summary_sentence(strengths: list[dict], weaknesses: list[dict]) -> str:
    if not strengths and not weaknesses:
        return "No metric clearly stands out or lags behind."
    parts = []
    if strengths:
        parts.append("Strong in: " + ", ".join(s["label"] for s in strengths) + ".")
    if weaknesses:
        parts.append("Weak in: " + ", ".join(w["label"] for w in weaknesses) + ".")
    return " ".join(parts)


def h2h_tactical_recommendations(
    our_fp: pd.Series,
    their_fp: pd.Series,
    our_name: str = "Our team",
    their_name: str = "Opponent",
    exploit_threshold: float = 25.0,
    danger_threshold: float = 25.0,
) -> dict:
    """Generate tactical suggestion texts from two teams' style fingerprints.

    Logic:
      - Our strength (>70 effective) + their weakness (<30 effective) → "exploit".
      - Our weakness + their strength → "beware".
      - Even metrics are noted.

    Args:
        our_fp, their_fp: percentile (0-100) Series; STYLE_FEATURES axis.

    Returns:
        {"exploit": [{metric, label, sentence}],
         "beware":  [{metric, label, sentence}],
         "neutral_notes": [str]}
    """
    exploit, beware, neutral = [], [], []

    common = [m for m in our_fp.index if m in their_fp.index]
    for metric in common:
        info = METRIC_LABELS.get(metric)
        if not info:
            continue
        label, direction = info

        ours = float(our_fp[metric])
        theirs = float(their_fp[metric])

        # Effective direction: high = positive for high_good metrics; low = positive for low_good ones.
        our_effective = ours if direction == "high_good" else (100 - ours)
        their_effective = theirs if direction == "high_good" else (100 - theirs)

        diff = our_effective - their_effective

        if our_effective >= 65 and their_effective <= (100 - exploit_threshold) and their_effective <= 40:
            sentence = f"✅ {label}: us {int(ours)}p, them {int(theirs)}p — press home this advantage."
            exploit.append({"metric": metric, "label": label, "ours": ours, "theirs": theirs, "sentence": sentence})
        elif our_effective <= 40 and their_effective >= 65:
            sentence = f"⚠️ {label}: them {int(theirs)}p, us {int(ours)}p — be ready for pressure from this side."
            beware.append({"metric": metric, "label": label, "ours": ours, "theirs": theirs, "sentence": sentence})
        elif abs(diff) < 10 and our_effective >= 55:
            neutral.append(f"{label}: both teams similar ({int(ours)} vs {int(theirs)}).")

    exploit.sort(key=lambda x: -abs(x["ours"] - x["theirs"]))
    beware.sort(key=lambda x: -abs(x["theirs"] - x["ours"]))

    return {
        "exploit": exploit[:5],
        "beware": beware[:5],
        "neutral_notes": neutral[:3],
        "our_name": our_name,
        "their_name": their_name,
    }


def style_fit_score(player_pct: pd.Series, target: pd.Series) -> dict:
    """Compute how well a player (role metrics, league %tile) fits a target style.

    Returns:
        {"fit_pct": 0-100, "matches": [{metric, label, player_pct, target}],
         "mismatches": [...]}
    """
    common = [m for m in target.index if m in player_pct.index]
    if not common:
        return {"fit_pct": 0.0, "matches": [], "mismatches": []}

    # Distance: mean absolute difference → 0 = perfect fit, 100 = complete opposite.
    diffs = [abs(float(player_pct[m]) - float(target[m])) for m in common]
    mean_diff = sum(diffs) / len(diffs)
    fit = max(0.0, 100.0 - mean_diff)

    matches, mismatches = [], []
    for m in common:
        d = abs(float(player_pct[m]) - float(target[m]))
        rec = {
            "metric": m,
            "label": METRIC_LABELS.get(m, (m, "high_good"))[0],
            "player_pct": float(player_pct[m]),
            "target": float(target[m]),
            "diff": d,
        }
        if d <= 20:
            matches.append(rec)
        elif d >= 35:
            mismatches.append(rec)

    matches.sort(key=lambda x: x["diff"])
    mismatches.sort(key=lambda x: -x["diff"])
    return {
        "fit_pct": round(fit, 1),
        "matches": matches[:5],
        "mismatches": mismatches[:5],
    }
