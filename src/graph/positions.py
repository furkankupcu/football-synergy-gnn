"""Pozisyon taksonomisi — FBref kısaltmaları + Transfermarkt detaylısı tek şemaya.

Hedef 12 pozisyon (kanat-spesifik): GK, RB, RCB, LCB, LB, RWB, LWB, CDM, CM,
CAM, RW, LW, ST. Aslında 13 var; RCB/LCB ayrı tutulur çünkü sol/sağ stoper
ikilisinin sinerjisi anlamlıdır.
"""

POSITION_TAXONOMY = [
    "GK", "RB", "RCB", "LCB", "LB", "RWB", "LWB",
    "CDM", "CM", "CAM", "RW", "LW", "ST",
]


# Transfermarkt main_position string'leri
TM_MAP = {
    "Goalkeeper": "GK",
    "Right-Back": "RB",
    "Left-Back": "LB",
    "Centre-Back": "CB",          # daha sonra LCB/RCB'ye dağıt
    "Right Midfield": "RW",
    "Left Midfield": "LW",
    "Right Winger": "RW",
    "Left Winger": "LW",
    "Defensive Midfield": "CDM",
    "Central Midfield": "CM",
    "Attacking Midfield": "CAM",
    "Second Striker": "CAM",
    "Centre-Forward": "ST",
}


# FBref kısaltmaları (genelde MF, FW, DF, GK + opsiyonel side)
# tek-harfli kodları da kabul ediyoruz: G/D/M/F
FBREF_MAP = {
    "GK": "GK",
    "DF": "CB",        # genel — yan bilgisi yoksa stoper varsay
    "MF": "CM",
    "FW": "ST",
    "DF,MF": "RWB",    # bek-orta saha hybrid
    "MF,DF": "CDM",
    "MF,FW": "CAM",
    "FW,MF": "CAM",
    "DF,FW": "RWB",
    "FW,DF": "RWB",
    # (player_stats.position) tek-harfli kodları
    "G": "GK",
    "D": "CB",
    "M": "CM",
    "F": "ST",
}


def normalize_position(tm_pos: str | None, fbref_pos: str | None, side_hint: str | None = None) -> str:
    """Tek bir oyuncu için en iyi tahmin pozisyonu döndürür.

    side_hint: 'R' / 'L' / None — TM 'Centre-Back' gibi yan-bilgisiz değerler için.
    """
    if tm_pos and tm_pos in TM_MAP:
        base = TM_MAP[tm_pos]
        if base == "CB":
            if side_hint == "L":
                return "LCB"
            if side_hint == "R":
                return "RCB"
            return "RCB"  # default — sezon başında dağılım eşitlenir
        return base

    if fbref_pos:
        key = fbref_pos.strip()
        if key in FBREF_MAP:
            base = FBREF_MAP[key]
            if base == "CB":
                return "RCB" if side_hint != "L" else "LCB"
            return base

    return "CM"  # düşürme: orta saha


def position_one_hot(pos: str) -> list[int]:
    return [1 if pos == p else 0 for p in POSITION_TAXONOMY]


def position_index(pos: str) -> int:
    try:
        return POSITION_TAXONOMY.index(pos)
    except ValueError:
        return POSITION_TAXONOMY.index("CM")


# --- 18-Zone System (6 rows × 3 cols) ---------------------------------------
# X (derinlik) 6 dilim: 0=kendi kale arkası, 5=rakip kale önü
# Y (yatay) 3 dilim: 0=sağ kanat, 1=merkez, 2=sol kanat
#
# Zone ID = row * 3 + col, 0..17
# Row eşikleri X için: 16.67 / 33.33 / 50 / 66.67 / 83.33
# Col eşikleri Y için: 33.33 / 66.67
# Eşikler oyuncu dağılımından kalibre — uniform 18-zone futbolda kötü çalışıyor
X_EDGES = [20.0, 37.0, 47.0, 54.0, 62.0]  # 6 row
Y_EDGES = [36.0, 65.0]                     # 3 col (sağ < 36 < merkez < 65 < sol)


def zone_of(avg_x: float | None, avg_y: float | None) -> int | None:
    if avg_x is None or avg_y is None:
        return None
    row = 0
    for i, e in enumerate(X_EDGES):
        if avg_x >= e:
            row = i + 1
    col = 0
    for i, e in enumerate(Y_EDGES):
        if avg_y >= e:
            col = i + 1
    return row * 3 + col


# Zone → kaba pozisyon kodu (ortada LCB/RCB için Y=50 alt-split fonksiyonda)
_ZONE_TO_POS_BASE: dict[int, str] = {
    # Row 0 (kendi kale önü, 0-16.67) — kaleci alanı
    0: "GK", 1: "GK", 2: "GK",
    # Row 1 (16.67-33.33) — defans hattı
    3: "RB", 4: "CB", 5: "LB",
    # Row 2 (33.33-50) — defans-orta geçiş
    6: "RB", 7: "CB", 8: "LB",
    # Row 3 (50-66.67) — orta saha
    9: "RW", 10: "CM", 11: "LW",
    # Row 4 (66.67-83.33) — hücum
    12: "RW", 13: "CAM", 14: "LW",
    # Row 5 (83.33-100) — forvet hattı
    15: "RW", 16: "ST", 17: "LW",
}


def coordinate_to_position(avg_x: float | None, avg_y: float | None,
                            base_code: str | None = None) -> str:
    """base_code (G/D/M/F) birincil; X/Y alt-pozisyon ayrımı için kullanılır.

    Hem Y (yatay) hem X (derinlik) eksenleri değerlendirilir:
      - D pozisyonu: X<38 (derin) → mutlaka stoper (RCB/LCB).
        Daha ileri X için Y kenarda ise RB/LB, orta ise üçlü-savunma içi stoper.
      - F pozisyonu: Y kenar → RW/LW, merkez → ST.
      - M pozisyonu: Y kenar → RW/LW; merkez ise X derinlik ile CDM/CM/CAM.

    Bu mantık Casemiro (M, X=45, Y=50) → CDM doğru kalır; ayrıca Smolčić
    gibi geride oynayan ama sol kenara yakın stoperler için LB değil LCB
    sonucunu verir.

    NOT: Tek-seferlik sezon ortalaması X/Y, Agyei gibi iki tarafta da oynayan
    oyuncularda yanıltıcı olabilir (ortalama Y merkeze düşer). Bu tip durumlar
    için `coordinate_to_position_voted()` kullanılmalıdır.
    """
    if base_code == "G":
        return "GK"

    # Koordinat yoksa: kaba fallback
    if avg_x is None or avg_y is None:
        return {"D": "RCB", "M": "CM", "F": "ST"}.get(base_code or "", "CM")

    x, y = float(avg_x), float(avg_y)
    Y_R, Y_L = Y_EDGES  # 36 sağ, 65 sol
    CB_DEPTH_THRESHOLD = 38.0  # bu X derinliğin altı kesin stoper

    # Defans (D)
    if base_code == "D":
        # Derin oynuyorsa kesinlikle stoper (LB/RB değil)
        if x < CB_DEPTH_THRESHOLD:
            return "RCB" if y < 50 else "LCB"
        # Daha ileri X — bek/stoper kararı için Y'ye bak
        if y < Y_R:
            return "RB"
        if y > Y_L:
            return "LB"
        # Merkez Y + ileri X → üçlü savunma içi stoper veya geç dönen bek
        return "RCB" if y < 50 else "LCB"

    # Forvet (F) — orta forvet vs kanat forvet
    if base_code == "F":
        if y < Y_R:
            return "RW"
        if y > Y_L:
            return "LW"
        return "ST"

    # Orta saha (M veya bilinmiyor) — derinlik + kanat ile alt-ayrım
    if y < Y_R:
        return "RW"
    if y > Y_L:
        return "LW"
    # Merkez Y → derinlik kararı
    if x < 48:
        return "CDM"
    if x < 57:
        return "CM"
    return "CAM"


def coordinate_to_position_voted(per_match_xy, base_code: str | None = None) -> str:
    """Her maç için ayrı pozisyon türet, en sık görülen kazanır (mode voting).

    Avantaj: Sezon boyunca sağda da solda da oynayan kanat oyuncusunun
    (örn. Agyei: 19 maç sağ, 6 maç sol) sezon ortalama Y'si merkeze
    düşer ve yanlışlıkla ST çıkar. Maç-bazlı türetim her oyunda doğru
    tarafı saptar; mod kazanır.

    Args:
        per_match_xy: her maç için (avg_x, avg_y) tuple/listesi.
        base_code: baz kodu (G/D/M/F).

    Returns:
        En sık görülen pozisyon kodu. Liste boşsa fallback.
    """
    from collections import Counter

    if not per_match_xy:
        return coordinate_to_position(None, None, base_code)

    positions = [
        coordinate_to_position(x, y, base_code)
        for x, y in per_match_xy
        if x is not None and y is not None
    ]
    if not positions:
        return coordinate_to_position(None, None, base_code)

    counts = Counter(positions)
    return counts.most_common(1)[0][0]


# Formasyon → pozisyon slot listesi (squad completion için)
FORMATIONS = {
    "4-3-3":   ["GK", "RB", "RCB", "LCB", "LB", "CDM", "CM", "CAM", "RW", "ST", "LW"],
    "4-2-3-1": ["GK", "RB", "RCB", "LCB", "LB", "CDM", "CDM", "CAM", "RW", "LW", "ST"],
    "3-5-2":   ["GK", "RCB", "RCB", "LCB", "RWB", "CDM", "CM", "CM", "LWB", "ST", "ST"],
    "4-4-2":   ["GK", "RB", "RCB", "LCB", "LB", "RW", "CM", "CM", "LW", "ST", "ST"],
}
