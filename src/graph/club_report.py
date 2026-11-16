"""Club report: a club's XI scored by the SynergyHGT model — weak links, transfer candidates,
and the squad's highest-synergy XI.

    python -m src.graph.club_report "Team name" [--lineup-season 2026-2027]
                                    [--include NAME ...] [--exclude NAME ...] [--out-dir DIR]

Two axes per outfield player (goalkeepers are not assessed), both as percentiles among all
≥900-minute players of the same position (nine leagues, model season):
  - synergy:     team fit with the other ten = mean predicted uplift with them
                 − mean predicted uplift with a random regular (removes "high with everyone")
  - performance: match rating as a z-score within the player's own league × position
need = 1 − mean(synergy pct, performance pct): high = weak link, transfer priority.

Writes to DIR (default reports/<team-slug>/): transfer_board.png, transfer_needs.png,
xi_most_used.png, xi_best_synergy.png, results.json; prints the tables.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import patheffects
from matplotlib.patches import FancyBboxPatch

from src import db
from src.graph.edge_builder import _season_where
from src.graph.recommend import SynergyEngine
from src.graph.xi import best_synergy_xi, most_used_xi
from src.graph.pitch_map import render

ap = argparse.ArgumentParser()
ap.add_argument("team")
ap.add_argument("--model-season", default="2025-2026")
ap.add_argument("--lineup-season", default=None, help="season of the XI (default: model season)")
ap.add_argument("--include", nargs="*", default=None, help="force into the XI (name fragment)")
ap.add_argument("--exclude", nargs="*", default=None, help="drop from the XI (name fragment)")
ap.add_argument("--out-dir", default=None)
ap.add_argument("--formation", default=None, help="shape of the highest-synergy XI (e.g. 4-3-3); default: roles of the most-used XI")
ap.add_argument("--min-minutes", type=int, default=90, help="squad players need this many minutes to enter the highest-synergy XI")
ARGS = ap.parse_args()
TEAM = ARGS.team
MODEL_SEASON = ARGS.model_season
LINEUP_SEASON = ARGS.lineup_season or MODEL_SEASON
MIN_MINUTES = 900
BUDGET_EUR, BUDGET_MAX_AGE = 15_000_000, 27
TOP_MAX_AGE = 32
SLUG = TEAM.lower().translate(str.maketrans("çğıöşü ", "cgiosu-"))
HERE = Path(ARGS.out_dir) if ARGS.out_dir else Path("reports") / SLUG
HERE.mkdir(parents=True, exist_ok=True)
LINEUP_TITLE, MODEL_TITLE = LINEUP_SEASON.replace("-20", "/"), MODEL_SEASON.replace("-20", "/")
TODAY = date(2026, 11, 16)

eng = SynergyEngine.load(MODEL_SEASON)
name = lambda i: eng.player_names[i]


def season_minutes(season: str, team: str | None = None) -> pd.DataFrame:
    where_sql, params = _season_where(season, None)
    extra = "AND" if where_sql else "WHERE"
    team_sql = f"{extra} p.team_name = :team" if team else ""
    return db.read_sql(f"""
        SELECT 'ss_' || p.source_player_id::text AS pid,
               SUM(p.minutes_played) AS minutes,
               SUM(p.rating * p.minutes_played) / NULLIF(SUM(CASE WHEN p.rating IS NOT NULL
                                                              THEN p.minutes_played END), 0) AS rating
        FROM player_stats p JOIN matches m ON m.match_id = p.match_id
        {where_sql} {team_sql}
        GROUP BY p.source_player_id
    """, {**params, "team": team} if team else params).set_index("pid")


KINDS = ["attack", "defence"] if eng.def_model is not None else ["attack"]


def sym(a: list[int], b: list[int], kind: str = "attack") -> torch.Tensor:
    """Symmetric predicted synergy for index pairs (a[k], b[k]); kind attack | defence."""
    return eng.pair_scores(a, b, kind)


# --- XI: most starts this season (shared with docs/gen_hero.py), role from this season --------
xi_df = most_used_xi(TEAM, LINEUP_SEASON, include=ARGS.include, exclude=ARGS.exclude)
xi = [eng.pid_to_idx[p] for p in xi_df["player_id"] if p in eng.pid_to_idx]
unknown = xi_df[~xi_df["player_id"].isin(eng.pid_to_idx)][["name", "role"]].to_dict("records")
role = {eng.pid_to_idx[p]: r for p, r in zip(xi_df["player_id"], xi_df["role"]) if p in eng.pid_to_idx}

# --- candidate pool: regulars of 2025/26, current club from player_details ------------------
stats = season_minutes(MODEL_SEASON)
details = db.read_sql("""
    SELECT 'ss_' || source_player_id::text AS pid, current_team_name, market_value, date_of_birth
    FROM player_details
""").set_index("pid")
pool = [eng.pid_to_idx[p] for p in stats.index
        if p in eng.pid_to_idx and stats.loc[p, "minutes"] >= MIN_MINUTES]
current_club = {i: details["current_team_name"].get(eng.player_ids[i]) or eng.teams[i] for i in pool + xi}

# Ratings are not comparable across leagues (weaker leagues / busy goalkeepers rate higher), so
# performance = rating z-score within the player's 2025/26 league × position, then percentile.
where_m, params_m = _season_where(MODEL_SEASON, None)
team_league = db.read_sql(f"""
    SELECT p.team_name AS team, MAX(m.league) AS league
    FROM player_stats p JOIN matches m ON m.match_id = p.match_id {where_m} GROUP BY p.team_name
""", params_m).set_index("team")["league"]
perf = pd.DataFrame({"idx": pool + xi})
perf["rating"] = [stats["rating"].get(eng.player_ids[i], np.nan) for i in perf["idx"]]
perf["league"] = [team_league.get(eng.teams[i]) for i in perf["idx"]]
perf["pos"] = [eng.positions[i] for i in perf["idx"]]
reg = perf[perf["idx"].isin(pool)]
grp = reg.groupby(["league", "pos"])["rating"].agg(["mean", "std"])
perf = perf.join(grp, on=["league", "pos"])
perf["z"] = (perf["rating"] - perf["mean"]) / perf["std"]
perf_z = dict(zip(perf["idx"], perf["z"]))
league_of = dict(zip(perf["idx"], perf["league"]))

gen = torch.Generator().manual_seed(0)
regulars = torch.tensor(pool)[torch.randperm(len(pool), generator=gen)[:300]].tolist()
base = {k: dict(zip(pool + xi, sym([c for c in pool + xi for _ in regulars], regulars * len(pool + xi),
                                   k).view(-1, len(regulars)).mean(1).tolist())) for k in KINDS}


def raw_rating(i: int) -> float | None:
    v = stats["rating"].get(eng.player_ids[i])
    return None if v is None or pd.isna(v) else round(float(v), 2)


def group(pos: str) -> set[str]:
    return {"RCB": {"RCB", "LCB"}, "LCB": {"LCB", "RCB"}}.get(pos, {pos})


def age(pid: str) -> int | None:
    dob = details["date_of_birth"].get(pid)
    return None if dob is None or pd.isna(dob) else (TODAY - dob).days // 365


def pct(values: np.ndarray, v: float) -> float:
    return float((values < v).mean())


rows, recs = [], {}
for i in xi:
    pos = role[i]
    if pos == "GK":
        continue  # goalkeepers are not assessed for now (but stay in the XI as teammates)
    others = [j for j in xi if j != i]
    cands = [c for c in pool if eng.positions[c] in group(pos) and c not in xi
             and current_club[c] != TEAM]
    # Team fit per kind = mean synergy with the other ten − mean synergy with a random regular,
    # then percentile among all candidates of the position. synergy = mean of attack and defence.
    kind_pct, me_kind = {}, {}
    for k in KINDS:
        fit = sym([c for c in cands + [i] for _ in others], others * (len(cands) + 1), k)
        tf = fit.view(-1, len(others)).mean(1).numpy() - np.array([base[k][c] for c in cands + [i]])
        kind_pct[k] = [pct(tf[:-1], v) for v in tf[:-1]]
        me_kind[k] = pct(tf[:-1], tf[-1])
    rating = np.array([perf_z.get(c, np.nan) for c in cands + [i]], dtype=float)  # league-relative
    perf_pool = rating[:-1][~np.isnan(rating[:-1])]

    me_syn = float(np.mean(list(me_kind.values())))
    me_perf = None if np.isnan(rating[-1]) else pct(perf_pool, rating[-1])
    parts = [p for p in (me_syn, me_perf) if p is not None]
    val = details["market_value"].get(eng.player_ids[i])
    rows.append({"player": name(i), "pos": pos, "club_2526": eng.teams[i],
                 "age": age(eng.player_ids[i]),
                 "value_m": None if val is None or pd.isna(val) else round(val / 1e6, 1),
                 "attack_pct": me_kind["attack"], "defence_pct": me_kind.get("defence"),
                 "synergy_pct": me_syn, "performance_pct": me_perf,
                 "rating_2526": raw_rating(i), "league_2526": league_of.get(i),
                 "need": round(1 - float(np.mean(parts)), 3)})

    cand_df = pd.DataFrame({
        "idx": cands, "rating": rating[:-1],
        "att_pct": kind_pct["attack"], "def_pct": kind_pct.get("defence", [np.nan] * len(cands)),
        "perf_pct": [np.nan if np.isnan(v) else pct(perf_pool, v) for v in rating[:-1]],
    })
    cand_df["syn_pct"] = cand_df[["att_pct", "def_pct"]].mean(axis=1)
    cand_df["score"] = cand_df[["syn_pct", "perf_pct"]].mean(axis=1)
    cand_df["value"] = [details["market_value"].get(eng.player_ids[c]) for c in cands]
    cand_df["age"] = [age(eng.player_ids[c]) for c in cands]
    me_score = 1 - (rows[-1]["need"] or 0)
    better = cand_df[cand_df["score"] > me_score]

    def fmt(df, n=5, key="score"):
        return [{"name": name(r.idx), "club": current_club[r.idx], "age": r.age,
                 "value_m": None if pd.isna(r.value) else round(r.value / 1e6, 1),
                 "attack_pct": round(r.att_pct, 2),
                 "defence_pct": None if pd.isna(r.def_pct) else round(r.def_pct, 2),
                 "synergy_pct": round(r.syn_pct, 2), "performance_pct": round(r.perf_pct, 2),
                 "rating": raw_rating(r.idx), "league_2526": league_of.get(r.idx)}
                for r in df.sort_values(key, ascending=False).head(n).itertuples()]

    budget = better[(better["value"] <= BUDGET_EUR) & (better["age"] <= BUDGET_MAX_AGE)]
    # "Best fit": highest synergy (attack + defence) with these ten, among realistic players
    # (performance at least the position median, age ≤ TOP_MAX_AGE)
    fit_pool = cand_df[(cand_df["perf_pct"] >= 0.5) & (cand_df["age"].fillna(99) <= TOP_MAX_AGE)]
    recs[name(i)] = {"best": fmt(better), "budget": fmt(budget, 12),
                     "best_fit": fmt(fit_pool, 12, key="syn_pct")}

needs = sorted(rows, key=lambda r: -(r["need"] or 0))

# --- pairs (for the report) -----------------------------------------------------------------
pairs = {k: sorted(((float(sym([a], [b], k)), name(a), name(b)) for a, b in combinations(xi, 2)),
                   reverse=True) for k in KINDS}

(HERE / "results.json").write_text(json.dumps(
    {"xi": xi_df[["name", "role", "starts", "minutes"]].to_dict("records"), "not_in_model": unknown, "needs": needs, "candidates": recs, "pairs": pairs},
    indent=2, ensure_ascii=False, default=str), encoding="utf-8")

# --- figure: synergy vs performance for the XI ----------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 7))
ax.axhspan(0, 0.5, xmin=0, xmax=0.5, color="#fee2e2", zorder=0)
ax.axhline(0.5, color="#94a3b8", lw=0.8); ax.axvline(0.5, color="#94a3b8", lw=0.8)
halo = [patheffects.withStroke(linewidth=3, foreground="white")]
placed: list[tuple[float, float]] = []
for r in sorted(rows, key=lambda r: -(r["performance_pct"] or 0)):
    if r["synergy_pct"] is None or r["performance_pct"] is None:
        continue
    x, y = r["synergy_pct"], r["performance_pct"]
    c = "#dc2626" if (r["need"] or 0) > 0.5 else "#2563eb"
    ax.scatter(x, y, s=120, color=c, zorder=3)
    # Clustered players (often several near the top right): stack labels downwards, leader lines
    dy = 0
    while any(abs(x - px) < 0.12 and abs(y + dy - py) < 0.035 for px, py in placed):
        dy -= 0.04
    placed.append((x, y + dy))
    right = x > 0.75  # label to the left of points near the right edge, else to the right
    ax.annotate(f"{r['player'].split()[-1]} ({r['pos']})", (x, y),
                xytext=(x - 0.025 if right else x + 0.025, y + dy),
                ha="right" if right else "left", va="center", fontsize=10, path_effects=halo,
                zorder=4, arrowprops=None if dy == 0 else {"arrowstyle": "-", "color": "#94a3b8", "lw": 0.7})
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel(f"{' + '.join(KINDS)} synergy with the rest of the XI  (percentile within position)")
ax.set_ylabel("individual performance 2025/26\n(match rating vs own league peers, percentile)")
ax.set_title(f"{TEAM} · where to strengthen\n{LINEUP_TITLE} XI · synergy from the {MODEL_TITLE} model",
             fontweight="bold", loc="left")
ax.text(0.02, 0.03, "weak on both\n→ transfer priority", color="#b91c1c", fontsize=9.5, va="bottom")
ax.text(0.98, 0.97, "strong on both", color="#1d4ed8", fontsize=9.5, ha="right", va="top")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "transfer_needs.png", dpi=150, bbox_inches="tight", facecolor="white")

# --- figure: transfer board — weakest positions, incumbent vs candidates ---------------------
BG, CARD, EDGE, TXT, SUB = "#0b1220", "#111a2e", "#1e293b", "#f1f5f9", "#94a3b8"
C_ATT, C_DEF, C_PERF, C_NOW = "#fb923c", "#34d399", "#60a5fa", "#f59e0b"
board = [r for r in needs if (r["need"] or 0) >= 0.15][:4] or needs[:3]


def card(ax, kind: str, p: dict, highlight: bool = False) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.04), 0.96, 0.92, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=CARD, ec=C_NOW if highlight else EDGE, lw=2 if highlight else 1))
    ax.text(0.08, 0.84, kind, color=C_NOW if highlight else SUB, fontsize=8, weight="bold")
    n = p["name"]
    ax.text(0.08, 0.68, n, color=TXT, weight="bold",  # long names shrink to stay inside the card
            fontsize=11.5 if len(n) <= 22 else 10 if len(n) <= 27 else 8.5)
    facts = [p.get("club"), f"{p['age']}" if p.get("age") else None,
             f"€{p['value_m']:g}M" if p.get("value_m") else None]
    ax.text(0.08, 0.555, " · ".join(f for f in facts if f), color=SUB, fontsize=8.5)
    if p.get("league"):
        ax.text(0.08, 0.455, f"2025/26: {p['league']}", color=SUB, fontsize=7.5, alpha=0.8)
    bars = [("Attack syn.", p["att"], C_ATT)]
    if "defence" in KINDS:  # only with a defensive model that carries signal (see EXPERIMENT_LOG)
        bars.append(("Defence syn.", p["def"], C_DEF))
    bars.append(("Performance", p["perf"], C_PERF))
    for y, (label, v, c) in zip((0.31, 0.195, 0.08)[-len(bars):], bars):
        ax.text(0.08, y + 0.035, label, color=SUB, fontsize=7.5, va="center")
        ax.add_patch(FancyBboxPatch((0.38, y), 0.44, 0.07, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=EDGE, ec="none"))
        if v is None or pd.isna(v):
            ax.text(0.6, y + 0.035, "n/a", color=SUB,
                    fontsize=7, ha="center", va="center")
            continue
        ax.add_patch(FancyBboxPatch((0.38, y), max(0.44 * v, 0.02), 0.07,
                                    boxstyle="round,pad=0,rounding_size=0.03", fc=c, ec="none"))
        ax.text(0.86, y + 0.035, f"{round(v * 100)}", color=TXT, fontsize=8.5, va="center",
                weight="bold")


n_rows, n_cols = len(board), 5
fig = plt.figure(figsize=(17, 2.55 * n_rows + 1.6), facecolor=BG)
gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[0.42] + [1] * n_cols,
                      left=0.01, right=0.99, top=1 - 1.15 / fig.get_figheight(), bottom=0.07,
                      wspace=0.06, hspace=0.12)
shown: set[str] = set()
for row, r in enumerate(board):
    lab = fig.add_subplot(gs[row, 0]); lab.axis("off")
    lab.text(0.5, 0.58, r["pos"], color=TXT, fontsize=22, weight="bold", ha="center", va="center")
    lab.text(0.5, 0.36, f"need {r['need']:.2f}", color=C_NOW if r["need"] >= 0.4 else SUB,
             fontsize=9, ha="center", va="center")
    me = {"name": r["player"], "club": TEAM, "age": r.get("age"), "value_m": r.get("value_m"),
          "league": r.get("league_2526"), "att": r["attack_pct"], "def": r["defence_pct"],
          "perf": r["performance_pct"]}
    card(fig.add_subplot(gs[row, 1]), "NOW", me, highlight=True)
    # 3 best fits (highest synergy with these ten) + 1 value pick; fresh names per row
    # (two CM rows would otherwise repeat the same players)
    fits = [c for c in recs[r["player"]]["best_fit"] if c["name"] not in shown][:3]
    shown.update(c["name"] for c in fits)
    value = [c for c in recs[r["player"]]["budget"] if c["name"] not in shown][:1]
    shown.update(c["name"] for c in value)
    picks = [(f"BEST FIT #{k}", c) for k, c in enumerate(fits, 1)] +             [(f"VALUE PICK · ≤€{BUDGET_EUR // 1_000_000}M, ≤{BUDGET_MAX_AGE}", c) for c in value]
    for col, (kind, c) in enumerate(picks[:4], start=2):
        card(fig.add_subplot(gs[row, col]), kind,
             {"name": c["name"], "club": c["club"], "age": c["age"], "value_m": c["value_m"],
              "league": c["league_2526"], "att": c["attack_pct"], "def": c["defence_pct"],
              "perf": c["performance_pct"]})
fig.text(0.01, 1 - 0.45 / fig.get_figheight(), f"{TEAM} · transfer board", color=TXT, fontsize=20,
         weight="bold", va="center")
fig.text(0.01, 1 - 0.85 / fig.get_figheight(),
         f"Weakest positions of the {LINEUP_TITLE} XI · best fits = highest {' + '.join(KINDS)} synergy with the other ten "
         f"(performance ≥ median) · synergy from the {MODEL_TITLE} SynergyHGT model", color=SUB, fontsize=10.5, va="center")
fig.text(0.01, 0.012,
         "Bars are percentiles among all ≥900-minute players of the same position in nine leagues (2025/26). "
         "Synergy: predicted fit with the other ten minus fit with a random regular — attack: goals + assists"
         + (", defence: xG conceded while together.\n" if "defence" in KINDS else
            " (defensive pair synergy is not measurable in this data).\n") +
         f"Performance: match rating relative to the player's own league and position. Top picks ≤ {TOP_MAX_AGE} years. "
         "Club, age and market value as of 2026-09.",
         color=SUB, fontsize=8.5)
fig.savefig(HERE / "transfer_board.png", dpi=150, facecolor=BG)

# --- most-used XI vs the squad's highest-synergy XI ------------------------------------------
lineup_title = LINEUP_SEASON.replace("-20", "/")
model_title = MODEL_SEASON.replace("-20", "/")
render(eng, xi_df, f"{TEAM} · {lineup_title} · most-used XI\nsynergy predicted by the {model_title} model",
       str(HERE / "xi_most_used.png"), MODEL_SEASON)
best_xi, s_used, s_best = best_synergy_xi(eng, TEAM, LINEUP_SEASON, xi_df, formation=ARGS.formation,
                                         min_minutes=ARGS.min_minutes)
# Compare as sets — with --formation the slot order differs from the most-used XI
used_ids, best_ids = set(xi_df["player_id"]), set(best_xi["player_id"])
players_in = best_xi[~best_xi["player_id"].isin(used_ids)]
players_out = xi_df[~xi_df["player_id"].isin(best_ids)]
shape = f" ({ARGS.formation})" if ARGS.formation else ""
render(eng, best_xi, f"{TEAM} · {lineup_title} squad · highest-synergy XI{shape}\n"
       f"total synergy {s_best:.1f} vs {s_used:.1f} for the most-used XI",
       str(HERE / "xi_best_synergy.png"), MODEL_SEASON,
       highlight={p for p in best_xi["player_id"] if p not in set(xi_df["player_id"])})
with open(HERE / "results.json", encoding="utf-8") as fh:
    res = json.load(fh)
res["best_synergy_xi"] = {"formation": ARGS.formation, "total_synergy": round(s_best, 3),
                          "most_used_total": round(s_used, 3),
                          "xi": best_xi[["name", "slot", "minutes"]].to_dict("records"),
                          "in": players_in["name"].tolist(), "out": players_out["name"].tolist()}
(HERE / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str),
                                   encoding="utf-8")

# --- console report -------------------------------------------------------------------------
print(f"HIGHEST-SYNERGY XI{shape}: total {s_best:.2f} vs {s_used:.2f} (most-used)")
print("  " + ", ".join(f"{r.slot} {r.name}" for r in best_xi.itertuples()))
print("  in:  " + ", ".join(f"{r.name} ({int(r.minutes)} min)" for r in players_in.itertuples()))
print("  out: " + ", ".join(players_out["name"]))
f = lambda v: "  –" if v is None else f"{v:.2f}"
print("NEEDS (weakest first)")
for r in needs:
    print(f"  {r['pos']:>3} {r['player']:<22} attack {f(r['attack_pct'])}  defence {f(r['defence_pct'])}  performance {f(r['performance_pct'])}"
          f"  rating {r['rating_2526']}  need {f(r['need'])}")
if unknown:
    print("NOT IN MODEL (no 2025/26 data in these leagues): "
          + ", ".join(f"{u['name']} ({u['role']})" for u in unknown))
for r in needs[:5]:
    print(f"\n{r['player']} ({r['pos']})")
    for key in ("best", "budget"):
        print(f"  {key}:")
        for c in recs[r["player"]][key][:4]:
            print(f"    {c['name']} ({c['club']}, {c['age']}, €{c['value_m']}M) att {c['attack_pct']} def {c['defence_pct']} "
                  f"perf {c['performance_pct']} rating {c['rating']} [{c['league_2526']}]")
