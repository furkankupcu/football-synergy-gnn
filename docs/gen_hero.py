"""README hero image: a team's most-used XI, edges = SynergyHGT predicted pair synergy.

Usage:  python -m docs.gen_hero [team] [season] [--lineup-season S] [--out PATH]
        (default: Arsenal 2025-2026 → docs/hero_synergy.png)

--lineup-season picks the XI and average positions from another season than the model's,
e.g. this season's squad (incl. new signings) scored by last season's model.
"""
from __future__ import annotations

import argparse
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, patheffects
from mplsoccer import VerticalPitch

from src import db
from src.graph.edge_builder import _season_where
from src.graph.positions import FORMATIONS
from src.graph.recommend import SynergyEngine

ap = argparse.ArgumentParser()
ap.add_argument("team", nargs="?", default="Arsenal")
ap.add_argument("season", nargs="?", default="2025-2026", help="model season")
ap.add_argument("--lineup-season", default=None, help="season for the XI and positions")
ap.add_argument("--out", default="docs/hero_synergy.png")
args = ap.parse_args()
TEAM, SEASON = args.team, args.season
LINEUP_SEASON = args.lineup_season or SEASON
TOP_EDGES = 12
# Slot → (length, width) on a 120×80 pitch, attacking upwards; width 80 = right touchline
XY = {"GK": (8, 40), "RB": (34, 70), "RCB": (28, 50), "LCB": (28, 30), "LB": (34, 10),
      "CDM": (50, 40), "CM": (66, 56), "CAM": (66, 24), "RW": (92, 68), "LW": (92, 12), "ST": (104, 40)}
BG, LINE, TXT, SUB = "#0b1220", "#334155", "#f1f5f9", "#94a3b8"

eng = SynergyEngine.load(SEASON)
where_sql, params = _season_where(LINEUP_SEASON, None)
extra = "AND" if where_sql else "WHERE"
mins = db.read_sql(f"""
    SELECT 'ss_' || p.source_player_id::text AS player_id, SUM(p.minutes_played) AS minutes
    FROM player_stats p JOIN matches m ON m.match_id = p.match_id
    {where_sql} {extra} p.team_name = :team
    GROUP BY p.source_player_id
""", {**params, "team": TEAM}).set_index("player_id")["minutes"]

# Pick the most-used players per line (a 6 labelled "CM" still beats a backup labelled "CDM"),
# then place them: exact position label first, leftovers fill the remaining slots of that line.
LINES = [["GK"], ["RB", "RCB", "LCB", "LB"], ["CDM", "CM", "CAM"], ["RW", "ST", "LW"]]
LINE_OF = {"GK": 0, "RB": 1, "RCB": 1, "LCB": 1, "LB": 1, "RWB": 1, "LWB": 1,
           "CDM": 2, "CM": 2, "CAM": 2, "RW": 3, "ST": 3, "LW": 3}
# Everyone who played for the team that season and exists in the model graph — new signings
# appear under their previous club in the graph, so match on minutes, not on graph team.
squad = sorted((eng.pid_to_idx[p] for p in mins.index if p in eng.pid_to_idx and mins[p] > 0),
               key=lambda i: -mins[eng.player_ids[i]])
xi: dict[str, int] = {}
for line, slots in enumerate(LINES):
    chosen = [i for i in squad if LINE_OF.get(eng.positions[i]) == line][:len(slots)]
    rest = []
    for i in chosen:  # by minutes: first player with a free matching slot takes it
        if eng.positions[i] in slots and eng.positions[i] not in xi:
            xi[eng.positions[i]] = i
        else:
            rest.append(i)
    for slot in slots:
        if slot not in xi and rest:
            xi[slot] = rest.pop(0)
assert len(xi) == 11, f"{TEAM}: could not fill the XI ({len(xi)}/11)"
xi = {s: xi[s] for line in LINES for s in line}

# Draw each player at his real season-average position (data: X 0→100 towards the opponent
# goal, Y 0 = right touchline); slot template only as fallback.
ids = ",".join(eng.player_ids[i][3:] for i in xi.values())  # numeric ids from the graph
avg = db.read_sql(f"""
    SELECT 'ss_' || a.source_player_id::text AS player_id,
           SUM(a.avg_x * a.points_count) / NULLIF(SUM(a.points_count), 0) AS x,
           SUM(a.avg_y * a.points_count) / NULLIF(SUM(a.points_count), 0) AS y
    FROM player_avg_positions a JOIN matches m ON m.match_id = a.match_id
    {where_sql} {extra} a.source_player_id IN ({ids})
    GROUP BY a.source_player_id
""", params).set_index("player_id")
for slot, i in xi.items():
    pid = eng.player_ids[i]
    if slot != "GK" and pid in avg.index and avg.loc[pid].notna().all():
        XY[slot] = (avg.loc[pid, "x"] * 1.2, (100 - avg.loc[pid, "y"]) * 0.8)

# Symmetric score: mean of both directions
pairs = sorted(
    ((eng._pair_score(a, [b]).item() + eng._pair_score(b, [a]).item()) / 2, sa, sb)
    for (sa, a), (sb, b) in combinations(xi.items(), 2)
)[::-1]
top = pairs[:TOP_EDGES]

pitch = VerticalPitch(pitch_type="statsbomb", pitch_color=BG, line_color=LINE, linewidth=1)
fig, ax = pitch.draw(figsize=(8, 10.5))
fig.patch.set_facecolor(BG)
norm = colors.Normalize(min(s for s, *_ in top), max(s for s, *_ in top))
cmap = plt.get_cmap("plasma")
for s, sa, sb in reversed(top):  # strongest drawn last (on top)
    (x1, y1), (x2, y2) = XY[sa], XY[sb]
    w = norm(s)
    pitch.lines(x1, y1, x2, y2, ax=ax, color=cmap(0.25 + 0.7 * w), lw=1.5 + 7 * w,
                alpha=0.45 + 0.5 * w, zorder=2)

halo = [patheffects.withStroke(linewidth=4, foreground=BG)]  # readable over lines
for slot, i in xi.items():
    x, y = XY[slot]
    pitch.scatter(x, y, ax=ax, s=850, color="#1e293b", edgecolors=TXT, linewidth=1.8, zorder=3)
    pitch.annotate(eng.positions[i], (x, y), ax=ax, ha="center", va="center", color=TXT,
                   fontsize=8, weight="bold", zorder=4)
    pitch.annotate(eng.player_names[i].split()[-1], (x - 5.8, y), ax=ax, ha="center",
                   va="center", color=TXT, fontsize=11, zorder=5, path_effects=halo)

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
fig.text(0.5, -0.085, f"Lines: the {TOP_EDGES} strongest pairs predicted by SynergyHGT. Players drawn at "
         "their season-average position.\nSynergy = output together minus what both players "
         "produce on their own.", ha="center", color=SUB, fontsize=10, linespacing=1.5)

fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=BG)
print(args.out, {s: eng.player_names[i] for s, i in xi.items()})
print([(round(s, 3), a, b) for s, a, b in top[:3]])
