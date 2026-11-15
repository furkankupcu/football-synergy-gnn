"""Otomatik takım/H2H taktiksel metin üretici — kural bazlı.

"Neden başarılı?" panelini ve H2H "nasıl kazanılır / nelere dikkat" çıktısını
besler. Hesaplama: stil parmak izinde (percentile) en yüksek/en düşük metrikler
ve iki takım arası asimetri.
"""

from __future__ import annotations

import pandas as pd


# Metrik adlandırması (Türkçe etiket + yorum yönü)
# direction: "high_good" → yüksek değer = pozitif yön
#            "low_good"  → düşük değer = pozitif yön (örn. xg_against, goals_against, foul)
METRIC_LABELS: dict[str, tuple[str, str]] = {
    "goals_for":               ("Atılan goller",            "high_good"),
    "goals_against":           ("Yenilen goller",           "low_good"),
    "xg":                      ("Beklenen gol (xG)",         "high_good"),
    "xg_against":              ("Yenilen xG",                "low_good"),
    "possession":              ("Top hakimiyeti",            "high_good"),
    "passes_pct":              ("Pas isabeti",               "high_good"),
    "shots":                   ("Şut sayısı",                "high_good"),
    "shots_on_target":         ("İsabetli şut",              "high_good"),
    "progressive_passes":      ("İlerleyici paslar",         "high_good"),
    "progressive_carries":     ("İlerleyici taşımalar",      "high_good"),
    "passes_into_final_third": ("Son üçlüye paslar",         "high_good"),
    "passes_into_penalty_area":("Ceza sahasına paslar",      "high_good"),
    "tackles":                 ("Müdahaleler",               "high_good"),
    "interceptions":           ("Top kapma",                 "high_good"),
    "blocks":                  ("Blok",                      "high_good"),
    "take_ons_attempted":      ("Dripling girişimi",         "high_good"),
    "take_ons_won":            ("Başarılı dripling",         "high_good"),
    "average_shot_distance":   ("Şut mesafesi",              "low_good"),
}


def team_strengths_weaknesses(
    fingerprint: pd.Series,
    high_threshold: float = 70.0,
    low_threshold: float = 30.0,
) -> dict:
    """Stil parmak izinden güçlü/zayıf yön ayrıştır.

    Args:
        fingerprint: pd.Series — feature_name → 0-100 lig percentile.

    Returns:
        {"strengths": [(metric, label, pct, sentence)],
         "weaknesses": [...],
         "summary": "kısa özet metni"}
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
            sentence = f"{label} ligin üst %{int(100 - effective_pct + (effective_pct))}'inde — güçlü yön."
            sentence = f"{label}: lig içi {int(pct)}. yüzdelik."
            strengths.append({"metric": metric, "label": label, "percentile": float(pct), "sentence": sentence})
        elif effective_pct <= low_threshold:
            sentence = f"{label}: lig içi {int(pct)}. yüzdelik — zayıf yön."
            weaknesses.append({"metric": metric, "label": label, "percentile": float(pct), "sentence": sentence})

    strengths.sort(key=lambda x: -x["percentile"] if METRIC_LABELS[x["metric"]][1] == "high_good" else x["percentile"])
    weaknesses.sort(key=lambda x: x["percentile"] if METRIC_LABELS[x["metric"]][1] == "high_good" else -x["percentile"])

    summary = _build_summary_sentence(strengths[:3], weaknesses[:3])
    return {"strengths": strengths[:6], "weaknesses": weaknesses[:6], "summary": summary}


def _build_summary_sentence(strengths: list[dict], weaknesses: list[dict]) -> str:
    if not strengths and not weaknesses:
        return "Belirgin bir öne çıkan/geride kalan metrik yok."
    parts = []
    if strengths:
        parts.append("Güçlü olduğu alanlar: " + ", ".join(s["label"] for s in strengths) + ".")
    if weaknesses:
        parts.append("Zayıf olduğu alanlar: " + ", ".join(w["label"] for w in weaknesses) + ".")
    return " ".join(parts)


def h2h_tactical_recommendations(
    our_fp: pd.Series,
    their_fp: pd.Series,
    our_name: str = "Bizim takım",
    their_name: str = "Rakip",
    exploit_threshold: float = 25.0,
    danger_threshold: float = 25.0,
) -> dict:
    """İki takım stil parmak izinden taktiksel öneri metni üretir.

    Mantık:
      - Bizim güçlü (>70 etkili) + onların zayıf (<30 etkili) metrik → "sömür".
      - Bizim zayıf + onların güçlü → "dikkat et".
      - Eşit metrikler not edilir.

    Args:
        our_fp, their_fp: percentile (0-100) Series; STYLE_FEATURES ekseni.

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

        # Etkili yön: yüksek-iyi metrikte yüksek = pozitif; düşük-iyi metrikte düşük = pozitif.
        our_effective = ours if direction == "high_good" else (100 - ours)
        their_effective = theirs if direction == "high_good" else (100 - theirs)

        diff = our_effective - their_effective

        if our_effective >= 65 and their_effective <= (100 - exploit_threshold) and their_effective <= 40:
            sentence = f"✅ {label}: biz {int(ours)}p, rakip {int(theirs)}p — bu yönden bastır."
            exploit.append({"metric": metric, "label": label, "ours": ours, "theirs": theirs, "sentence": sentence})
        elif our_effective <= 40 and their_effective >= 65:
            sentence = f"⚠️ {label}: rakip {int(theirs)}p, biz {int(ours)}p — bu yönden gelen baskıya hazırlıklı ol."
            beware.append({"metric": metric, "label": label, "ours": ours, "theirs": theirs, "sentence": sentence})
        elif abs(diff) < 10 and our_effective >= 55:
            neutral.append(f"{label}: iki takım benzer ({int(ours)} vs {int(theirs)}).")

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
    """Bir oyuncunun (rol metrikleri lig %tile) hedef stile uyumunu hesaplar.

    Returns:
        {"fit_pct": 0-100, "matches": [{metric, label, player_pct, target}],
         "mismatches": [...]}
    """
    common = [m for m in target.index if m in player_pct.index]
    if not common:
        return {"fit_pct": 0.0, "matches": [], "mismatches": []}

    # Distance: ortalama mutlak fark → 0 = mükemmel uyum, 100 = tamamen ters.
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
