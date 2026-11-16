"""Position taxonomy — FBref abbreviations + detailed Transfermarkt positions in one scheme.

Target: 12 side-specific positions: GK, RB, RCB, LCB, LB, RWB, LWB, CDM, CM,
CAM, RW, LW, ST. There are actually 13; RCB/LCB are kept apart because the synergy of a
left/right centre-back pairing is meaningful.
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
    "Centre-Back": "CB",          # split into LCB/RCB later
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


# FBref abbreviations (usually MF, FW, DF, GK + optional side)
# single-letter codes are accepted too: G/D/M/F
FBREF_MAP = {
    "GK": "GK",
    "DF": "CB",        # generic — assume centre-back when there is no side information
    "MF": "CM",
    "FW": "ST",
    "DF,MF": "RWB",    # bek-orta saha hybrid
    "MF,DF": "CDM",
    "MF,FW": "CAM",
    "FW,MF": "CAM",
    "DF,FW": "RWB",
    "FW,DF": "RWB",
    # (player_stats.position) single-letter codes
    "G": "GK",
    "D": "CB",
    "M": "CM",
    "F": "ST",
}


def normalize_position(tm_pos: str | None, fbref_pos: str | None, side_hint: str | None = None) -> str:
    """Return the best-guess position for a single player.

    side_hint: 'R' / 'L' / None — for side-less values such as TM 'Centre-Back'.
    """
    if tm_pos and tm_pos in TM_MAP:
        base = TM_MAP[tm_pos]
        if base == "CB":
            if side_hint == "L":
                return "LCB"
            if side_hint == "R":
                return "RCB"
            return "RCB"  # default — balanced out across the season
        return base

    if fbref_pos:
        key = fbref_pos.strip()
        if key in FBREF_MAP:
            base = FBREF_MAP[key]
            if base == "CB":
                return "RCB" if side_hint != "L" else "LCB"
            return base

    return "CM"  # fallback: midfield


def position_one_hot(pos: str) -> list[int]:
    return [1 if pos == p else 0 for p in POSITION_TAXONOMY]


def position_index(pos: str) -> int:
    try:
        return POSITION_TAXONOMY.index(pos)
    except ValueError:
        return POSITION_TAXONOMY.index("CM")


# --- 18-Zone System (6 rows × 3 cols) ---------------------------------------
# X (depth) 6 bands: 0 = behind own goal, 5 = in front of the opponent goal
# Y (width) 3 bands: 0 = right wing, 1 = centre, 2 = left wing
#
# Zone ID = row * 3 + col, 0..17
# Row thresholds for X: 16.67 / 33.33 / 50 / 66.67 / 83.33
# Column thresholds for Y: 33.33 / 66.67
# Thresholds calibrated on the player distribution — a uniform 18-zone grid works poorly in football
X_EDGES = [20.0, 37.0, 47.0, 54.0, 62.0]  # 6 row
Y_EDGES = [36.0, 65.0]                     # 3 columns (right < 36 < centre < 65 < left)


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


# Zone → coarse position code (LCB/RCB in the centre are split at Y=50 in the function)
_ZONE_TO_POS_BASE: dict[int, str] = {
    # Row 0 (in front of own goal, 0-16.67) — goalkeeper area
    0: "GK", 1: "GK", 2: "GK",
    # Row 1 (16.67-33.33) — defensive line
    3: "RB", 4: "CB", 5: "LB",
    # Row 2 (33.33-50) — defence-to-midfield
    6: "RB", 7: "CB", 8: "LB",
    # Row 3 (50-66.67) — orta saha
    9: "RW", 10: "CM", 11: "LW",
    # Row 4 (66.67-83.33) — attack
    12: "RW", 13: "CAM", 14: "LW",
    # Row 5 (83.33-100) — forward line
    15: "RW", 16: "ST", 17: "LW",
}


def coordinate_to_position(avg_x: float | None, avg_y: float | None,
                            base_code: str | None = None) -> str:
    """base_code (G/D/M/F) is primary; X/Y only splits it into sub-positions.

    Both the Y (width) and X (depth) axes are used:
      - D: X<38 (deep) → always a centre-back (RCB/LCB).
        Further up, an edge Y gives RB/LB, a central Y a centre-back of a back three.
      - F: edge Y → RW/LW, centre → ST.
      - M: edge Y → RW/LW; centre → CDM/CM/CAM by X depth.

    With this rule Casemiro (M, X=45, Y=50) correctly stays CDM, and deep centre-backs
    close to the left touchline such as Smolčić come out as LCB
    rather than LB.

    NOTE: a single season-average X/Y can mislead for players who play on both sides
    such as Agyei (the average Y falls in the centre). Use
    `coordinate_to_position_voted()` for such cases.
    """
    if base_code == "G":
        return "GK"

    # No coordinates: coarse fallback
    if avg_x is None or avg_y is None:
        return {"D": "RCB", "M": "CM", "F": "ST"}.get(base_code or "", "CM")

    x, y = float(avg_x), float(avg_y)
    Y_R, Y_L = Y_EDGES  # 36 right, 65 left
    CB_DEPTH_THRESHOLD = 38.0  # below this X depth: always a centre-back
    # Centre-backs of high-line teams sit at X≈38-42, Y≈30-35/65-70
    # (Saliba, Guéhi, Schlotterbeck); the X<38 rule made them full-backs. Real full-backs
    # at this depth spread to Y<28 / Y>72. Fixed 77 defenders in 25/26 (2026-11-16).
    CB_HIGH_LINE_X, CB_HIGH_LINE_Y = 43.0, (28.0, 72.0)

    # Defans (D)
    if base_code == "D":
        # Playing deep → always a centre-back (not LB/RB)
        if x < CB_DEPTH_THRESHOLD or (
            x < CB_HIGH_LINE_X and CB_HIGH_LINE_Y[0] <= y <= CB_HIGH_LINE_Y[1]
        ):
            return "RCB" if y < 50 else "LCB"
        # Further up X — use Y to decide full-back vs centre-back
        if y < Y_R:
            return "RB"
        if y > Y_L:
            return "LB"
        # Central Y + higher X → centre-back of a back three or an inverted full-back
        return "RCB" if y < 50 else "LCB"

    # Forvet (F) — orta forvet vs kanat forvet
    if base_code == "F":
        if y < Y_R:
            return "RW"
        if y > Y_L:
            return "LW"
        return "ST"

    # Midfield (M or unknown) — split by depth + flank
    if y < Y_R:
        return "RW"
    if y > Y_L:
        return "LW"
    # Central Y → decide by depth
    if x < 48:
        return "CDM"
    if x < 57:
        return "CM"
    return "CAM"


def coordinate_to_position_voted(per_match_xy, base_code: str | None = None) -> str:
    """Derive a position per match; the most frequent one wins (mode voting).

    Advantage: a winger who plays on both the right and the left during a season
    (e.g. Agyei: 19 matches right, 6 left) has a season-average Y in the centre
    and wrongly comes out as ST. Per-match derivation finds the right side in every
    match; the mode wins.

    Args:
        per_match_xy: list/tuple of (avg_x, avg_y) per match.
        base_code: base code (G/D/M/F).

    Returns:
        The most frequent position code. Fallback if the list is empty.
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


# Formation → list of position slots (for squad completion)
FORMATIONS = {
    "4-3-3":   ["GK", "RB", "RCB", "LCB", "LB", "CDM", "CM", "CAM", "RW", "ST", "LW"],
    "4-2-3-1": ["GK", "RB", "RCB", "LCB", "LB", "CDM", "CDM", "CAM", "RW", "LW", "ST"],
    "3-5-2":   ["GK", "RCB", "RCB", "LCB", "RWB", "CDM", "CM", "CM", "LWB", "ST", "ST"],
    "4-4-2":   ["GK", "RB", "RCB", "LCB", "LB", "RW", "CM", "CM", "LW", "ST", "ST"],
}
