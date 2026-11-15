"""Oyuncu benzerliği — embedding cosine similarity üzerinden.

İkinci hedef ("X'in yerine geçebilecek oyuncular") için iskelet.
"""

from __future__ import annotations

import torch
from src.graph.recommend import SynergyEngine, Recommendation


def find_similar(engine: SynergyEngine, player_id: str, k: int = 10,
                 same_position: bool = True) -> list[Recommendation]:
    idx = engine.pid_to_idx.get(player_id)
    if idx is None:
        raise KeyError(f"Bilinmeyen player_id: {player_id}")

    emb = engine._emb
    q = emb[idx]
    sims = torch.nn.functional.cosine_similarity(q.unsqueeze(0), emb)
    sims[idx] = -1.0  # kendini hariç tut

    if same_position:
        target_pos = engine.positions[idx]
        mask = torch.tensor([p == target_pos for p in engine.positions], dtype=torch.bool)
        sims = sims.masked_fill(~mask, -1.0)

    top = torch.topk(sims, k=min(k, mask.sum().item() if same_position else len(sims)))
    return [
        Recommendation(
            player_id=engine.player_ids[i.item()],
            player_name=engine.player_names[i.item()],
            position=engine.positions[i.item()],
            team=engine.teams[i.item()],
            score=float(s.item()),
        )
        for s, i in zip(top.values, top.indices) if s.item() > -1.0
    ]
