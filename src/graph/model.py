"""SynergyHGT — Heterogeneous Graph Transformer + edge regressor.

Predicts the target (uplift or joint) from the embeddings of two players.

Phase 1.4 addition: learned team and league embeddings are concatenated with the input
features. The "same team" / "same league" bias is learned in a separate channel, so the
remaining synergy embedding is cleaned of it.

Cold-start support: a new player gets an embedding from node features + team/league
embeddings alone (even without any message-passing edge).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData


class SynergyHGT(nn.Module):
    def __init__(
        self,
        metadata: tuple,
        in_dim: int,
        hidden: int = 128,
        out: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_teams: int | None = None,
        num_leagues: int | None = None,
        team_emb_dim: int = 8,
        league_emb_dim: int = 4,
    ):
        super().__init__()
        self.use_team = num_teams is not None and num_teams > 0
        self.use_league = num_leagues is not None and num_leagues > 0
        if self.use_team:
            self.team_emb = nn.Embedding(num_teams, team_emb_dim)
        if self.use_league:
            self.league_emb = nn.Embedding(num_leagues, league_emb_dim)

        self.lin_in = nn.ModuleDict({
            node_type: Linear(-1, hidden) for node_type in metadata[0]
        })
        self.convs = nn.ModuleList([
            HGTConv(hidden, hidden, metadata, heads=num_heads)
            for _ in range(num_layers)
        ])
        self.lin_out = Linear(hidden, out)

        # Edge regressor: [emb_a | emb_b | |emb_a - emb_b| | emb_a*emb_b] → scalar
        self.edge_mlp = nn.Sequential(
            nn.Linear(4 * out, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def _augment(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Player x'ine team/league embedding'lerini concat eder."""
        x_dict = {nt: data[nt].x for nt in data.node_types}
        if "player" in x_dict:
            parts = [x_dict["player"]]
            if self.use_team and hasattr(data["player"], "team_idx"):
                parts.append(self.team_emb(data["player"].team_idx))
            if self.use_league and hasattr(data["player"], "league_idx"):
                parts.append(self.league_emb(data["player"].league_idx))
            x_dict["player"] = torch.cat(parts, dim=-1)
        return x_dict

    def encode(self, data: HeteroData) -> dict[str, torch.Tensor]:
        x_dict = self._augment(data)
        x_dict = {nt: self.lin_in[nt](x) for nt, x in x_dict.items()}
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: F.gelu(v) for k, v in x_dict.items()}
        x_dict = {k: self.lin_out(v) for k, v in x_dict.items()}
        return x_dict

    def predict_edge(self, emb: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        ea, eb = emb[src], emb[dst]
        feat = torch.cat([ea, eb, (ea - eb).abs(), ea * eb], dim=-1)
        return self.edge_mlp(feat).squeeze(-1)

    def forward(self, data: HeteroData, edge_pairs: torch.Tensor) -> torch.Tensor:
        """edge_pairs: [2, N] — predicted target (uplift/joint)."""
        emb = self.encode(data)["player"]
        return self.predict_edge(emb, edge_pairs[0], edge_pairs[1])
