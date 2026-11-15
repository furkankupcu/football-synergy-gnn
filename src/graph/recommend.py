"""Sinerji-tabanlı kadro tamamlama inference.

Kullanım:
    eng = SynergyEngine.load(season="2024-2025")
    recs = eng.recommend_squad(
        formation="4-3-3",
        filled={"GK": "thibaut_courtois", "LW": "vinicius_junior"},
        top_k=5,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
from torch_geometric.data import HeteroData

from src.graph.model import SynergyHGT
from src.graph.positions import FORMATIONS

GRAPH_DIR = Path("data/graph")
MODELS_DIR = Path("data/graph/models")


@dataclass
class Recommendation:
    player_id: str
    player_name: str
    position: str
    team: str
    score: float  # toplam sinerji skoru


class SynergyEngine:
    def __init__(self, data: HeteroData, model: SynergyHGT, device: str = "cpu"):
        self.data = data.to(device)
        self.model = model.to(device).eval()
        self.device = device
        self.player_ids: list[str] = data["player"].player_id
        self.player_names: list[str] = data["player"].player_name
        self.positions: list[str] = data["player"].position
        self.teams: list[str] = data["player"].team
        self.pid_to_idx = {pid: i for i, pid in enumerate(self.player_ids)}

        with torch.no_grad():
            self._emb = self.model.encode(self.data)["player"]

    @classmethod
    def load(cls, season: str, device: str = "cpu") -> "SynergyEngine":
        data: HeteroData = torch.load(GRAPH_DIR / f"hetero_{season}.pt", weights_only=False)
        ckpt = torch.load(MODELS_DIR / f"synergy_hgt_{season}.pt", weights_only=False)
        sd = ckpt["state_dict"]
        num_teams = sd["team_emb.weight"].shape[0] if "team_emb.weight" in sd else None
        num_leagues = sd["league_emb.weight"].shape[0] if "league_emb.weight" in sd else None
        model = SynergyHGT(
            metadata=data.metadata(),
            in_dim=data["player"].x.shape[1],
            hidden=ckpt.get("hidden", 128),
            out=ckpt.get("embed_dim", 64),
            num_teams=num_teams,
            num_leagues=num_leagues,
            team_emb_dim=sd["team_emb.weight"].shape[1] if num_teams else 8,
            league_emb_dim=sd["league_emb.weight"].shape[1] if num_leagues else 4,
        )
        model.load_state_dict(sd)
        return cls(data, model, device=device)

    # Pozisyon "aileleri" — esnek modda aynı ailede çapraz öneri için
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

    def _candidates_for_position(self, pos: str, exclude: set[str],
                                  flexible: bool = False) -> list[int]:
        allowed = self.POS_FAMILY.get(pos, {pos}) if flexible else {pos}
        return [i for i, p in enumerate(self.positions)
                if p in allowed and self.player_ids[i] not in exclude]

    def _pair_score(self, a_idx: int, b_indices: list[int]) -> torch.Tensor:
        if not b_indices:
            return torch.zeros(1, device=self.device)
        src = torch.full((len(b_indices),), a_idx, dtype=torch.long, device=self.device)
        dst = torch.tensor(b_indices, dtype=torch.long, device=self.device)
        with torch.no_grad():
            return self.model.predict_edge(self._emb, src, dst)

    def recommend_squad(
        self,
        formation: str | list[str],
        filled: dict[str, str],
        top_k: int = 5,
        flexible_positions: bool = False,
    ) -> dict[str, list[Recommendation]]:
        if isinstance(formation, list):
            slots = formation
        else:
            slots = FORMATIONS.get(formation)
            if slots is None:
                raise ValueError(f"Bilinmeyen formasyon: {formation}. Seçenekler: {list(FORMATIONS)}")

        filled_idx = [self.pid_to_idx[pid] for pid in filled.values() if pid in self.pid_to_idx]
        used = set(filled.values())

        # Pozisyon başına slot sayısını say (örn 3-5-2'de iki ST)
        empty_slots: dict[str, int] = {}
        for s in slots:
            if s in filled:
                continue
            empty_slots[s] = empty_slots.get(s, 0) + 1

        recs: dict[str, list[Recommendation]] = {}
        for pos, n_slots in empty_slots.items():
            candidates = self._candidates_for_position(
                pos, exclude=used, flexible=flexible_positions)
            if not candidates:
                recs[pos] = []
                continue

            with torch.no_grad():
                if filled_idx:
                    src = torch.tensor(candidates, dtype=torch.long, device=self.device).repeat_interleave(len(filled_idx))
                    dst = torch.tensor(filled_idx, dtype=torch.long, device=self.device).repeat(len(candidates))
                    scores = self.model.predict_edge(self._emb, src, dst).view(len(candidates), len(filled_idx))
                    total = scores.sum(dim=1)
                else:
                    # Dolu oyuncu yoksa — node embedding norm'una göre default sıralama
                    total = self._emb[candidates].norm(dim=1)

            top = torch.topk(total, k=min(top_k * n_slots, len(candidates)))
            recs[pos] = [
                Recommendation(
                    player_id=self.player_ids[candidates[i.item()]],
                    player_name=self.player_names[candidates[i.item()]],
                    position=pos,
                    team=self.teams[candidates[i.item()]],
                    score=float(s.item()),
                )
                for s, i in zip(top.values, top.indices)
            ]
        return recs

    def player_embedding(self, player_id: str) -> torch.Tensor | None:
        idx = self.pid_to_idx.get(player_id)
        return self._emb[idx] if idx is not None else None
