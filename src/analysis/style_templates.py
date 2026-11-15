"""Önceden tanımlı oyun stili şablonları.

Her şablon, `src.analysis.style.STYLE_FEATURES` listesindeki metrikler için
hedef percentile (0-100) değerleri içerir. Bir takımın bu şablona uyumu,
`style_fingerprint` çıktısı ile cosine similarity hesaplanarak ölçülür.

Şablon kullanımı:
    >>> from src.analysis.style_templates import STYLE_TEMPLATES, get_template_vector
    >>> target = get_template_vector("high-press")
    >>> # target: pd.Series indeksli (feature_name → 0-100 percentile)
"""

from __future__ import annotations

import pandas as pd

from src.analysis.style import STYLE_FEATURES


# Her şablon: STYLE_FEATURES listesindeki kolonlar için hedef percentile (0-100).
# Yüksek değer = "bu metrikte ligin üst yüzdelerinde olmak istiyoruz".
# Eksik metrikler 50 (nötr) sayılır.
STYLE_TEMPLATES: dict[str, dict] = {
    "high-press": {
        "label": "Yüksek Pres / Gegenpressing",
        "description": "Top kaybı sonrası anında geri kazanım, yüksek defansif aksiyon, kısa pas hakimiyeti.",
        "targets": {
            "tackles": 85, "interceptions": 85, "blocks": 75,
            "take_ons_won": 70, "take_ons_attempted": 75,
            "passes_pct": 75, "possession": 70,
            "progressive_passes": 70, "progressive_carries": 70,
            "passes_into_final_third": 75, "passes_into_penalty_area": 70,
            "shots": 70, "shots_on_target": 70,
            "xg": 75, "xg_against": 25,
            "goals_for": 75, "goals_against": 25,
            "average_shot_distance": 30,  # yakın şutlar
        },
    },
    "possession": {
        "label": "Pozisyon Oyunu (Tiki-Taka)",
        "description": "Yüksek top hakimiyeti, kısa-doğru paslar, sabırlı saha kazanımı.",
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
        "label": "Direkt / Kontra-Atak",
        "description": "Düşük blok, hızlı geçişler, dikey paslar, az dokunuşla hücum.",
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
        "label": "Defansif Blok",
        "description": "Kompakt savunma, düşük top hakimiyeti, kalan az gole odaklı yapı.",
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
        "label": "Açık Oyun / Yüksek Skor",
        "description": "İki tarafa açık karakter, çok şut/xG ama defansif kırılganlık.",
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
    """API için: ID + label + description listesi."""
    return [
        {"id": tid, "label": t["label"], "description": t["description"]}
        for tid, t in STYLE_TEMPLATES.items()
    ]


def get_template_vector(template_id: str) -> pd.Series:
    """Şablon ID → STYLE_FEATURES sırasıyla hedef percentile Series'i.

    Eksik metrikler 50 (nötr) doldurulur.
    """
    if template_id not in STYLE_TEMPLATES:
        raise ValueError(f"Bilinmeyen şablon: {template_id}")
    tgt = STYLE_TEMPLATES[template_id]["targets"]
    return pd.Series({f: float(tgt.get(f, 50)) for f in STYLE_FEATURES})


def parse_style_target(style_target: str | None, team_stats: pd.DataFrame | None = None) -> pd.Series | None:
    """Genel parser: 'template:high-press' veya 'team:PSG' formatlarını çözer.

    None döndürürse hiçbir stil hedefi seçilmemiştir.
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
