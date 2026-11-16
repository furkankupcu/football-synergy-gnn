"""README hero image: a team's most-used XI, edges = SynergyHGT predicted pair synergy.

Usage:  python -m docs.gen_hero [team] [season] [--lineup-season S] [--include NAME ...]
                                [--exclude NAME ...] [--out PATH]
        (default: Arsenal 2025-2026 → docs/hero_synergy.png)

--lineup-season picks the XI and average positions from another season than the model's,
e.g. this season's squad (incl. new signings) scored by last season's model.
--include / --exclude override the automatic XI (most starts) by name fragment.
Players the model has never seen (not in the training season's graph) are drawn grey.
"""
from __future__ import annotations

import argparse
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, patheffects
import numpy as np
from mplsoccer import VerticalPitch

from src.graph.recommend import SynergyEngine
from src.graph.xi import most_used_xi

ap = argparse.ArgumentParser()
ap.add_argument("team", nargs="?", default="Arsenal")
ap.add_argument("season", nargs="?", default="2025-2026", help="model season")
ap.add_argument("--lineup-season", default=None, help="season for the XI and positions")
ap.add_argument("--include", nargs="*", default=None)
ap.add_argument("--exclude", nargs="*", default=None)
ap.add_argument("--out", default="docs/hero_synergy.png")
args = ap.parse_args()
TEAM, SEASON = args.team, args.season
LINEUP_SEASON = args.lineup_season or SEASON
TOP_EDGES = 12
BG, LINE, TXT, SUB, GREY = "#0b1220", "#334155", "#f1f5f9", "#94a3b8", "#475569"

eng = SynergyEngine.load(SEASON)
xi = most_used_xi(TEAM, LINEUP_SEASON, include=args.include, exclude=args.exclude)
idx = [eng.pid_to_idx.get(p) for p in xi["player_id"]]  # None = player unknown to the model

# Pitch coords: 120 long × 80 wide, attacking upwards; data y 0 = right touchline → width 80.
# GK (and anyone without position data) at a fixed spot.
xy = {}
for r in xi.itertuples():
    if r.role != "GK" and not (np.isnan(r.x) or np.isnan(r.y)):
        xy[r.Index] = np.array([r.x * 1.2, (100 - r.y) * 0.8])
    else:
        xy[r.Index] = np.array([8.0, 40.0] if r.role == "GK" else [60.0, 40.0])

# Players with nearly the same average spot would hide each other: push overlapping nodes
# apart until they are at least MIN_GAP pitch units apart.
MIN_GAP = 9.0
for _ in range(60):
    for a, b in combinations(xy, 2):
        d = xy[b] - xy[a]
        dist = float(np.hypot(*d))
        if dist < MIN_GAP:
            step = (d / dist if dist > 1e-6 else np.array([0.0, 1.0])) * (MIN_GAP - dist) / 2
            xy[a] -= step; xy[b] += step
xy = {k: np.clip(v, [2, 2], [118, 78]) for k, v in xy.items()}

# Symmetric score (mean of both directions), only between players the model knows
known = [k for k, i in enumerate(idx) if i is not None]
pairs = sorted(
    ((eng._pair_score(idx[a], [idx[b]]).item() + eng._pair_score(idx[b], [idx[a]]).item()) / 2, a, b)
    for a, b in combinations(known, 2)
)[::-1]
top = pairs[:TOP_EDGES]

pitch = VerticalPitch(pitch_type="statsbomb", pitch_color=BG, line_color=LINE, linewidth=1)
fig, ax = pitch.draw(figsize=(8, 10.5))
fig.patch.set_facecolor(BG)
norm = colors.Normalize(min(s for s, *_ in top), max(s for s, *_ in top))
cmap = plt.get_cmap("plasma")
for s, a, b in reversed(top):  # strongest drawn last (on top)
    (x1, y1), (x2, y2) = xy[a], xy[b]
    w = norm(s)
    pitch.lines(x1, y1, x2, y2, ax=ax, color=cmap(0.25 + 0.7 * w), lw=1.5 + 7 * w,
                alpha=0.45 + 0.5 * w, zorder=2)

halo = [patheffects.withStroke(linewidth=4, foreground=BG)]  # readable over lines
for r in xi.itertuples():
    x, y = xy[r.Index]
    seen = idx[r.Index] is not None
    pitch.scatter(x, y, ax=ax, s=850, color="#1e293b" if seen else BG,
                  edgecolors=TXT if seen else GREY, linewidth=1.8, zorder=3,
                  linestyle="-" if seen else "--")
    pitch.annotate(r.role, (x, y), ax=ax, ha="center", va="center", color=TXT if seen else GREY,
                   fontsize=8, weight="bold", zorder=4)
    pitch.annotate(r.name.split()[-1], (x - 5.8, y), ax=ax, ha="center", va="center",
                   color=TXT if seen else GREY, fontsize=11, zorder=5, path_effects=halo)

title = f"{TEAM} · {LINEUP_SEASON.replace('-20', '/')} season · most-used XI"
if LINEUP_SEASON != SEASON:
    title += f"\nsynergy predicted by the {SEASON.replace('-20', '/')} model"
ax.set_title(title, color=TXT, fontsize=17, weight="bold", pad=12)
sm = plt.cm.ScalarMappable(cmap=colors.ListedColormap(cmap([0.25 + 0.7 * t / 255 for t in range(256)])),
                           norm=norm)
cax = fig.add_axes([0.3, -0.005, 0.4, 0.012])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_label("predicted synergy (uplift per 90)", color=SUB, fontsize=9.5)
cb.ax.tick_params(colors=SUB, labelsize=8.5); cb.outline.set_visible(False)
note = (f"Lines: the {TOP_EDGES} strongest pairs predicted by SynergyHGT. Players drawn at "
        "their season-average position.\nSynergy = output together minus what both players "
        "produce on their own.")
if len(known) < len(xi):
    note += f"\nDashed grey: not in the {SEASON.replace('-20', '/')} data the model knows (no prediction)."
fig.text(0.5, -0.085 - 0.02 * (len(known) < len(xi)), note, ha="center", color=SUB, fontsize=10,
         linespacing=1.5)

fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=BG)
print(args.out, dict(zip(xi["role"], xi["name"])), f"{len(known)}/11 in model")
print([(round(s, 3), xi.name[a], xi.name[b]) for s, a, b in top[:3]])
