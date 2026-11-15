"""Transfer analizi — eksik mevki tespiti, oyuncu benzerliği, değer/performans oranı."""

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
    """Ham mevki stringini standart gruba eşler."""
    if not isinstance(pos, str):
        return "Unknown"
    pos_upper = pos.upper().strip()
    # FBref bazen 'MF,FW' gibi çoklu mevki verir → ilkini al
    primary = pos_upper.split(",")[0].split("/")[0].strip()
    for group, variants in POSITION_GROUPS.items():
        if primary in variants:
            return group
        if any(v in primary for v in variants):
            return group
    return "Unknown"


def squad_position_gaps(team_players: pd.DataFrame) -> pd.DataFrame:
    """Kadrodaki mevki dağılımını analiz eder, eksik mevzileri döndürür."""
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
            "status": "EKSİK" if current < req else "TAMAM",
        })
    return pd.DataFrame(rows)


def build_player_feature_matrix(player_stats: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Oyuncu feature matrisi (per-90 normalize edilmiş)."""
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
    """Cosine similarity ile hedef oyuncuya en benzer n oyuncuyu döndürür."""
    df = player_stats.copy()
    if "minutes" in df.columns:
        df = df[pd.to_numeric(df["minutes"], errors="coerce").fillna(0) >= min_minutes]

    if "player" not in df.columns or target_player not in df["player"].values:
        raise ValueError(f"Oyuncu bulunamadı veya yeterli dakikası yok: {target_player}")

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
    """Piyasa değeri / üretim (xG+xA per90) oranı — yüksek = ucuz, üretken."""
    df = player_stats.copy()
    if "minutes" in df.columns:
        df = df[pd.to_numeric(df["minutes"], errors="coerce").fillna(0) >= min_minutes]

    xg = pd.to_numeric(df.get("xg", 0), errors="coerce").fillna(0)
    xa = pd.to_numeric(df.get("xa", 0), errors="coerce").fillna(0)
    minutes = pd.to_numeric(df.get("minutes", np.nan), errors="coerce")
    df["xg_xa_per90"] = (xg + xa) / minutes * 90

    if market_value_col in df.columns:
        mv = pd.to_numeric(df[market_value_col], errors="coerce")
        # Düşük değer + yüksek üretim = yüksek skor
        df["value_score"] = df["xg_xa_per90"] / (mv / 1e6).replace(0, np.nan)
    return df.sort_values("value_score", ascending=False, na_position="last")


# ─── Stil uyum & kadro değerlendirme ─────────────────────────────────────────

# Formasyon → slot listesi: her slot bir (slot_id, label, pozisyon_kodu, suggested_role).
# pozisyon_kodu: 11-pozisyon taxonomy (GNN sistemi ile uyumlu).
# suggested_role: POSITION_PROFILES anahtarı — slot için stil uyumunu hesaplarken kullanılır.
FORMATION_SLOTS: dict[str, list[tuple[str, str, str, str]]] = {
    "4-3-3": [
        ("gk",   "Kaleci",     "GK",  "Goalkeeper"),
        ("lb",   "Sol Bek",    "LB",  "Full-Back"),
        ("lcb",  "Sol Stoper", "LCB", "Ball-Playing CB"),
        ("rcb",  "Sağ Stoper", "RCB", "Combative CB / Stopper"),
        ("rb",   "Sağ Bek",    "RB",  "Full-Back"),
        ("dm",   "Ön Libero",  "CDM", "Defensive Midfielder (#6)"),
        ("lcm",  "Sol O.Saha", "CM",  "Box-to-Box Midfielder"),
        ("rcm",  "Sağ O.Saha", "CM",  "Attacking Midfielder (#8)"),
        ("lw",   "Sol Kanat",  "LW",  "Classic Winger"),
        ("st",   "Santrafor",  "ST",  "Poacher"),
        ("rw",   "Sağ Kanat",  "RW",  "Inverted Winger"),
    ],
    "4-2-3-1": [
        ("gk",   "Kaleci",         "GK",  "Goalkeeper"),
        ("lb",   "Sol Bek",        "LB",  "Full-Back"),
        ("lcb",  "Sol Stoper",     "LCB", "Ball-Playing CB"),
        ("rcb",  "Sağ Stoper",     "RCB", "Combative CB / Stopper"),
        ("rb",   "Sağ Bek",        "RB",  "Full-Back"),
        ("ldm",  "Sol Ön Libero",  "CDM", "Defensive Midfielder (#6)"),
        ("rdm",  "Sağ Ön Libero",  "CDM", "Box-to-Box Midfielder"),
        ("lam",  "Sol O.Saha",     "LW",  "Attacking Midfielder (#8)"),
        ("cam",  "10 Numara",      "CAM", "Playmaker (#10)"),
        ("ram",  "Sağ O.Saha",     "RW",  "Attacking Midfielder (#8)"),
        ("st",   "Santrafor",      "ST",  "Pressing Forward"),
    ],
    "3-5-2": [
        ("gk",   "Kaleci",     "GK",  "Goalkeeper"),
        ("lcb",  "Sol Stoper", "LCB", "Wide CB (in 3)"),
        ("ccb",  "Libero",     "RCB", "Libero / Middle CB"),
        ("rcb",  "Sağ Stoper", "RCB", "Wide CB (in 3)"),
        ("lwb",  "Sol Kanat",  "LB",  "Wing-Back"),
        ("rwb",  "Sağ Kanat",  "RB",  "Wing-Back"),
        ("dm",   "Ön Libero",  "CDM", "Deep-Lying Playmaker"),
        ("lcm",  "Sol O.Saha", "CM",  "Box-to-Box Midfielder"),
        ("rcm",  "Sağ O.Saha", "CM",  "Attacking Midfielder (#8)"),
        ("ls",   "Sol Forvet", "ST",  "Target Man (#9)"),
        ("rs",   "Sağ Forvet", "ST",  "Poacher"),
    ],
    "4-4-2": [
        ("gk",   "Kaleci",     "GK",  "Goalkeeper"),
        ("lb",   "Sol Bek",    "LB",  "Full-Back"),
        ("lcb",  "Sol Stoper", "LCB", "Combative CB / Stopper"),
        ("rcb",  "Sağ Stoper", "RCB", "Combative CB / Stopper"),
        ("rb",   "Sağ Bek",    "RB",  "Full-Back"),
        ("lm",   "Sol Kanat",  "LW",  "Classic Winger"),
        ("lcm",  "Sol O.Saha", "CM",  "Box-to-Box Midfielder"),
        ("rcm",  "Sağ O.Saha", "CM",  "Deep-Lying Playmaker"),
        ("rm",   "Sağ Kanat",  "RW",  "Classic Winger"),
        ("ls",   "Sol Forvet", "ST",  "Target Man (#9)"),
        ("rs",   "Sağ Forvet", "ST",  "Poacher"),
    ],
}


def get_formation_slots(formation: str) -> list[dict]:
    """Formasyon ID → API'ye uygun slot listesi (dict formatında)."""
    raw = FORMATION_SLOTS.get(formation, FORMATION_SLOTS["4-3-3"])
    return [
        {"slot_id": sid, "label": lbl, "position_code": pos, "suggested_role": role}
        for (sid, lbl, pos, role) in raw
    ]


def compute_style_fit(
    player_metrics: dict[str, float],
    target_vector: dict[str, float] | "pd.Series",
) -> float:
    """Oyuncu metrik percentile dict'i ile hedef vektör arasındaki uyum (0-100).

    Hesap: ortak metriklerde ortalama mutlak fark, 100'den çıkarılır.
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
    """Her slot için yerleşmiş oyuncunun stile uyumunu + (boşsa) öneri ihtiyacını döndürür.

    Args:
        team_players: oyuncu havuzu (mevcut takım); minimum kolonlar:
                      source_player_id, player_name, position + STYLE_FEATURES
                      ile aynı isimde percentile kolonları DEĞİL — burada metrik
                      kolonları beklenir, percentile hesaplanır (lig içi rank).
        formation: "4-3-3" vb.
        target_vector: STYLE_FEATURES eksenli percentile Series. None ise uyum yok.
        slot_assignments: {slot_id: player_id?} — kullanıcının yerleştirdiği oyuncular.

    Returns:
        Slot başına dict listesi: {slot_id, label, position_code, suggested_role,
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
                    # Oyuncunun rol-metrik percentile'ını lig içinde hesapla (yaklaşım).
                    # Pratik: target_vector index'leri TEAM-level STYLE_FEATURES; oyuncuda
                    # bunlar 1:1 yok. Bunun yerine kaba bir proxy kullan:
                    #   player_metrics[m] = p[m] varsa o, yoksa 50.
                    p_metrics = {m: float(p[m]) if m in p and pd.notna(p[m]) else 50.0
                                 for m in target_vector.index}
                    rec["fit_pct"] = compute_style_fit(p_metrics, target_vector)
                    if rec["fit_pct"] >= 70:
                        rec["reason"] = "Stile yüksek uyum."
                    elif rec["fit_pct"] >= 50:
                        rec["reason"] = "Sınırda uyum — bazı metriklerde eksik."
                    else:
                        rec["reason"] = "Stile uymuyor — rotasyon/satış adayı."
        out.append(rec)
    return out


def radar_data(player_stats: pd.DataFrame, players: list[str]) -> pd.DataFrame:
    """Seçili oyuncular için lig içi percentile DataFrame'i döndürür (radar için)."""
    X, features = build_player_feature_matrix(player_stats)
    percentiles = (X.rank(pct=True) * 100).round(1)
    mask = player_stats["player"].isin(players)
    result = percentiles[mask.values].copy()
    result.index = player_stats.loc[mask, "player"].values
    return result[features]
