"""Performans-ağırlıklı transfer önerisi — Tactical Lab için.

Var olan SynergyEngine'i (`src.graph.recommend`) genişletmek yerine sarıyoruz:
1. Önce GNN squad-recommend ile pozisyona uygun aday listesini al (esnek pos aile dahil).
2. Aday havuzu üzerinde 4 bileşenli combined score hesapla:
   - style_fit       (style_fit modülünden, per-aday)
   - role_score      (oyuncunun rol-pct × rating, kabaca lig içi 0-100)
   - gnn_synergy     (yerleşik kadrodaki yerleşik oyuncularla ortalama predict_edge)
   - position_match  (slot=adayın_pos: 1.0; aynı aile: 0.7; başka: 0.3)
3. min_minutes hard-filter ile dakika eşiğini geçemeyenleri eler (Batuhan Şen sorunu).

Çıktı, frontend'in 'replacements' kart listesi için doğrudan tüketilebilir.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.graph.recommend import SynergyEngine


# Pozisyon ailesi — SynergyEngine.POS_FAMILY ile aynı mantık, burada local'e koyup
# import bağımlılığını sade tutuyoruz.
POS_FAMILY: dict[str, set[str]] = {
    "GK":  {"GK"},
    "RB":  {"RB", "RWB", "RCB"},
    "LB":  {"LB", "LWB", "LCB"},
    "RCB": {"RCB", "LCB", "CDM", "RB"},
    "LCB": {"LCB", "RCB", "CDM", "LB"},
    "CDM": {"CDM", "CM", "RCB", "LCB"},
    "CM":  {"CM", "CDM", "CAM"},
    "CAM": {"CAM", "CM", "RW", "LW", "ST"},
    "RW":  {"RW", "LW", "CAM", "ST", "RB"},
    "LW":  {"LW", "RW", "CAM", "ST", "LB"},
    "ST":  {"ST", "CAM", "RW", "LW"},
}


def position_match(slot: str, cand_pos: str) -> float:
    if not slot or not cand_pos:
        return 0.3
    s, c = slot.upper(), cand_pos.upper()
    if s == c:
        return 1.0
    if c in POS_FAMILY.get(s, set()):
        return 0.7
    return 0.3


def gnn_synergy_avg(
    eng: SynergyEngine,
    cand_pid: str,
    placed_pids: list[str],
) -> Optional[float]:
    """Adayın, yerleşik oyuncularla ortalama predict_edge skoru. None: aday graf'ta yok."""
    ai = eng.pid_to_idx.get(cand_pid)
    if ai is None or not placed_pids:
        return None
    b_indices = [eng.pid_to_idx[p] for p in placed_pids if p in eng.pid_to_idx]
    if not b_indices:
        return None
    with torch.no_grad():
        src = torch.full((len(b_indices),), ai, dtype=torch.long, device=eng.device)
        dst = torch.tensor(b_indices, dtype=torch.long, device=eng.device)
        scores = eng.model.predict_edge(eng._emb, src, dst)
    return float(scores.mean().item())


def rank_replacements(
    eng: SynergyEngine,
    slot: str,
    lineup: dict[str, str],            # current slot → pid
    candidate_pool: pd.DataFrame,      # cols: source_player_id, position, team_name, minutes, rating, role_score, style_fit
    weights: dict | None = None,
    top_k: int = 5,
    exclude_pids: set[str] | None = None,
) -> list[dict]:
    """Slot için en iyi 5 değişimi combined score'a göre döndürür.

    candidate_pool: önceden filtrelenmiş aday havuzu (min_minutes, position family vs).
    """
    w = {"position": 0.20, "style_fit": 0.30, "perf": 0.25, "synergy": 0.25}
    if weights:
        w.update(weights)

    placed = [p for p in lineup.values() if p]
    excl = set(exclude_pids or [])
    rows = []
    for r in candidate_pool.itertuples():
        pid = f"ss_{int(r.source_player_id)}"
        if pid in excl or pid in placed:
            continue
        pos_score = position_match(slot, getattr(r, "position", "") or "")
        style = float(getattr(r, "style_fit", 0.0) or 0.0) / 100.0
        perf = float(getattr(r, "role_score", 0.0) or 0.0) / 100.0
        syn = gnn_synergy_avg(eng, pid, placed)
        syn_n = max(0.0, min(1.0, (syn or 0.0) / 1.2))
        combined = (
            w["position"] * pos_score +
            w["style_fit"] * style +
            w["perf"] * perf +
            w["synergy"] * syn_n
        )
        rows.append({
            "pid": pid,
            "name": getattr(r, "player_name", "?"),
            "team": getattr(r, "team_name", "?"),
            "position": getattr(r, "position", None),
            "combined_score": round(combined, 4),
            "components": {
                "position": round(pos_score, 3),
                "style_fit": round(style, 3),
                "perf": round(perf, 3),
                "synergy": round(syn_n, 3),
            },
            "raw": {
                "minutes": int(getattr(r, "minutes", 0) or 0),
                "rating":  float(getattr(r, "rating", 0) or 0),
                "gnn_synergy_avg": round(syn, 4) if syn is not None else None,
            },
        })

    rows.sort(key=lambda x: -x["combined_score"])
    return rows[:top_k]


def build_why_text(rec: dict, top_contributing: list[dict] | None = None) -> str:
    """'Neden bu oyuncu' kısa açıklama."""
    comps = rec["components"]
    best = max(comps.items(), key=lambda kv: kv[1])
    label = {"position": "pozisyon eşleşmesi", "style_fit": "stil uyumu",
             "perf": "lig içi performans", "synergy": "kadro sinerjisi"}.get(best[0], best[0])
    extra = ""
    if top_contributing:
        metrics = ", ".join(c["metric"] for c in top_contributing[:2])
        extra = f" Güçlü metrikler: {metrics}."
    return f"En yüksek katkı: {label} ({best[1]:.0%}).{extra}"
