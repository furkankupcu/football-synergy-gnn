"""Pitch map of a team's most-used XI, edges = SynergyHGT predicted pair synergy
(command-line wrapper around src.graph.pitch_map.render).

Usage:  python -m docs.gen_hero [team] [season] [--lineup-season S] [--include NAME ...]
                                [--exclude NAME ...] [--out PATH]
        (default: Paris Saint-Germain 2025-2026 → pitch_map.png)

--lineup-season picks the XI and average positions from another season than the model's,
e.g. this season's squad (incl. new signings) scored by last season's model.
--include / --exclude override the automatic XI (most starts) by name fragment.
Players the model has never seen (not in the training season's graph) are drawn grey.
"""
from __future__ import annotations

import argparse

from src.graph.pitch_map import render
from src.graph.recommend import SynergyEngine
from src.graph.xi import most_used_xi


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("team", nargs="?", default="Paris Saint-Germain")
    ap.add_argument("season", nargs="?", default="2025-2026", help="model season")
    ap.add_argument("--lineup-season", default=None, help="season for the XI and positions")
    ap.add_argument("--include", nargs="*", default=None)
    ap.add_argument("--exclude", nargs="*", default=None)
    ap.add_argument("--out", default="pitch_map.png")
    args = ap.parse_args()
    lineup_season = args.lineup_season or args.season
    eng = SynergyEngine.load(args.season)
    xi = most_used_xi(args.team, lineup_season, include=args.include, exclude=args.exclude)
    title = f"{args.team} · {lineup_season.replace('-20', '/')} season · most-used XI"
    if lineup_season != args.season:
        title += f"\nsynergy predicted by the {args.season.replace('-20', '/')} model"
    top = render(eng, xi, title, args.out, args.season)
    print(args.out, dict(zip(xi["role"], xi["name"])), top[:3])
