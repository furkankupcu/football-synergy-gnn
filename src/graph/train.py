"""SynergyHGT training.

Split: player-disjoint 70/15/15 (default) or random edge split.
Loss: MSE (optionally minutes-weighted) + optional pairwise ranking loss.
Metrics: MSE, Spearman ρ, cold-start ρ.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from scipy.stats import spearmanr

from src.graph.model import SynergyHGT


def _ranking_loss(pred: torch.Tensor, target: torch.Tensor,
                  num_pairs: int = 8192, gen: torch.Generator | None = None) -> torch.Tensor:
    """Pairwise logistic ranking loss — Spearman surrogate.

    Samples random (i, j) pairs; where target[i] != target[j] it rewards the
    prediction for getting the order right: softplus(-sign(Δy)·Δpred).
    Unlike MSE it optimises the RANKING (ρ) directly.
    """
    n = pred.shape[0]
    if n < 2:
        return pred.new_tensor(0.0)
    dev = pred.device
    i = torch.randint(0, n, (num_pairs,), generator=gen, device=dev)
    j = torch.randint(0, n, (num_pairs,), generator=gen, device=dev)
    dy = target[i] - target[j]
    mask = dy.abs() > 1e-6
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    s = torch.sign(dy[mask])
    dp = pred[i][mask] - pred[j][mask]
    return F.softplus(-s * dp).mean()

GRAPH_DIR = Path("data/graph")
MODELS_DIR = Path("data/graph/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _split_edges(num_edges: int, train: float = 0.7, val: float = 0.15, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_edges, generator=g)
    n_train = int(num_edges * train)
    n_val = int(num_edges * val)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def _split_edges_player_disjoint(
    pair_src: torch.Tensor,
    pair_dst: torch.Tensor,
    num_players: int,
    train: float = 0.7,
    val: float = 0.15,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Split players, not edges.

    train edge: her iki ucu da train_player
    val edge:   her iki ucu train∪val ve en az biri val (inductive)
    test edge:  en az bir ucu test_player

    This measures real generalisation: test players' teammate edges never appear in
    training or in message passing (training edges are train-train only).
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_players, generator=g)
    n_train_p = int(num_players * train)
    n_val_p = int(num_players * val)
    train_players = perm[:n_train_p]
    val_players = perm[n_train_p:n_train_p + n_val_p]
    test_players = perm[n_train_p + n_val_p:]

    is_train = torch.zeros(num_players, dtype=torch.bool); is_train[train_players] = True
    is_val   = torch.zeros(num_players, dtype=torch.bool); is_val[val_players] = True
    is_test  = torch.zeros(num_players, dtype=torch.bool); is_test[test_players] = True

    src_train, dst_train = is_train[pair_src], is_train[pair_dst]
    src_val,   dst_val   = is_val[pair_src],   is_val[pair_dst]
    src_test,  dst_test  = is_test[pair_src],  is_test[pair_dst]

    train_mask = src_train & dst_train
    test_mask  = (src_test | dst_test)
    val_mask   = ((src_val | dst_val) & ~test_mask & (src_train | src_val) & (dst_train | dst_val))

    train_idx = train_mask.nonzero(as_tuple=False).squeeze(-1)
    val_idx   = val_mask.nonzero(as_tuple=False).squeeze(-1)
    test_idx  = test_mask.nonzero(as_tuple=False).squeeze(-1)

    info = {
        "players": {"train": int(is_train.sum()), "val": int(is_val.sum()), "test": int(is_test.sum())},
        "edges":   {"train": int(train_mask.sum()), "val": int(val_mask.sum()), "test": int(test_mask.sum())},
    }
    return train_idx, val_idx, test_idx, info


def _build_message_passing_data(
    data: HeteroData,
    train_idx: torch.Tensor,
    edge_dropout: float = 0.0,
    generator: torch.Generator | None = None,
) -> HeteroData:
    """Use only training teammate edges for message passing (prevents leakage).

    `build.py` stores every edge twice: first half original, second half reversed,
    so the training indices are applied to both halves.

    With edge_dropout > 0 that share of training edges is masked on every call,
    both directions of an edge together. Regularises cold-start embeddings.
    """
    out = data.clone()
    et = ("player", "teammate", "player")
    ei = data[et].edge_index
    half = ei.shape[1] // 2
    train_mask = torch.zeros(half, dtype=torch.bool)
    train_mask[train_idx] = True

    if edge_dropout > 0.0:
        keep = torch.ones(half, dtype=torch.bool)
        rand = torch.rand(half, generator=generator) if generator is not None else torch.rand(half)
        drop_in_train = (rand < edge_dropout) & train_mask
        keep[drop_in_train] = False
        train_mask = train_mask & keep

    full_mask = torch.cat([train_mask, train_mask], dim=0)
    out[et].edge_index = ei[:, full_mask]
    if hasattr(data[et], "edge_attr"):
        out[et].edge_attr = data[et].edge_attr[full_mask]
    return out


def _cold_start_mask(
    pair_src: torch.Tensor,
    pair_dst: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
) -> torch.Tensor:
    """Test edges whose **two** players never appear in a training edge.

    True cold-start: both players enter the model only through their node features
    (no teammate edge). This is how the model performs on genuinely new signings.
    """
    seen = torch.zeros(int(max(pair_src.max(), pair_dst.max())) + 1, dtype=torch.bool)
    seen[pair_src[train_idx]] = True
    seen[pair_dst[train_idx]] = True
    src_cold = ~seen[pair_src[test_idx]]
    dst_cold = ~seen[pair_dst[test_idx]]
    return src_cold & dst_cold


def train(season: str, epochs: int = 100, lr: float = 1e-3, hidden: int = 128, embed_dim: int = 64,
          patience: int = 15, device: str = "cpu", split: str = "player-disjoint",
          edge_dropout: float = 0.0, weighted: bool = False, rank_weight: float = 0.0,
          weight_decay: float = 1e-4, seed: int = 42, target: str = "uplift") -> None:
    """target: 'uplift' (attacking synergy, G+A) or 'def' (defensive synergy, xG conceded
    together). Each target is a separate model: synergy_hgt[_def]_<season>.pt."""
    # Model-init seed only; the player split keeps its own fixed seed so runs stay comparable
    torch.manual_seed(seed)
    np.random.seed(seed)

    pt = GRAPH_DIR / f"hetero_{season}.pt"
    data: HeteroData = torch.load(pt, weights_only=False)
    data = data.to(device)

    et = ("player", "teammate", "player")
    # Edges are stored in both directions → one direction is enough for labels
    ei_full = data[et].edge_index
    half = ei_full.shape[1] // 2
    pair_src = ei_full[0, :half]
    pair_dst = ei_full[1, :half]
    if target == "def" and not hasattr(data[et], "y_def"):
        raise RuntimeError("graph has no y_def — rebuild edges (edge_builder) and graph (build)")
    y = (data[et].y_def if target == "def" else data[et].y)[:half]
    suffix = "_def" if target == "def" else ""

    # Per-edge reliability weight: sqrt(minutes_together).
    # edge_attr column order (build.py): [minutes_together, matches_together, joint_g,
    #   joint_a, joint_xg, joint_xa, per90_joint_output]
    if hasattr(data[et], "edge_attr"):
        minutes_together = data[et].edge_attr[:half, 0].clamp(min=1.0)
        edge_w = minutes_together.sqrt()
    else:
        edge_w = torch.ones(half, device=device)

    split_info = None
    if split == "player-disjoint":
        num_players = data["player"].x.shape[0]
        train_idx, val_idx, test_idx, split_info = _split_edges_player_disjoint(
            pair_src.cpu(), pair_dst.cpu(), num_players
        )
        train_idx = train_idx.to(device); val_idx = val_idx.to(device); test_idx = test_idx.to(device)
        # Edges without a label (def: no shot data) still pass messages but get no loss/metric
        has_y = torch.isfinite(y)
        train_idx, val_idx, test_idx = (i[has_y[i]] for i in (train_idx, val_idx, test_idx))
        print(f"[split] player-disjoint | players={split_info['players']} | edges={split_info['edges']}")
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            raise RuntimeError(f"player-disjoint split produced an empty partition: {split_info}")
    elif split == "random":
        train_idx, val_idx, test_idx = _split_edges(half)
        print(f"[split] random (edge-level) | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    else:
        raise ValueError(f"bilinmeyen split: {split} (player-disjoint|random)")

    num_teams = int(getattr(data["player"], "num_teams", 0))
    num_leagues = int(getattr(data["player"], "num_leagues", 0))
    model = SynergyHGT(metadata=data.metadata(), in_dim=data["player"].x.shape[1],
                       hidden=hidden, out=embed_dim,
                       num_teams=num_teams or None,
                       num_leagues=num_leagues or None).to(device)
    print(f"[model] team_emb={'on' if num_teams else 'off'} ({num_teams}) | "
          f"league_emb={'on' if num_leagues else 'off'} ({num_leagues})")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_rho = -float("inf")
    patience_left = patience
    best_state = None
    dropout_gen = torch.Generator().manual_seed(0)
    rank_gen = torch.Generator(device=device).manual_seed(1)
    w_train = edge_w[train_idx]
    history: list[dict] = []
    print(f"[train] edge_dropout={edge_dropout} weighted={weighted} "
          f"rank_weight={rank_weight} wd={weight_decay}")

    for epoch in range(1, epochs + 1):
        model.train()
        mp_data = _build_message_passing_data(data, train_idx,
                                              edge_dropout=edge_dropout,
                                              generator=dropout_gen)
        emb = model.encode(mp_data)["player"]
        preds_train = model.predict_edge(emb, pair_src[train_idx], pair_dst[train_idx])
        # MSE (optionally minutes-weighted: down-weight noisy low-minute pairs)
        if weighted:
            se = (preds_train - y[train_idx]) ** 2
            loss = (w_train * se).sum() / w_train.sum()
        else:
            loss = loss_fn(preds_train, y[train_idx])
        # Ranking term (Spearman surrogate) — improves the ordering directly
        if rank_weight > 0:
            loss = loss + rank_weight * _ranking_loss(
                preds_train, y[train_idx], gen=rank_gen)
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            # Eval: no dropout, full training graph
            mp_eval = _build_message_passing_data(data, train_idx, edge_dropout=0.0)
            emb = model.encode(mp_eval)["player"]
            preds_val = model.predict_edge(emb, pair_src[val_idx], pair_dst[val_idx])
            val_mse = loss_fn(preds_val, y[val_idx]).item()
            rho = spearmanr(preds_val.cpu().numpy(), y[val_idx].cpu().numpy()).correlation if len(val_idx) > 5 else 0.0

        print(f"epoch {epoch:3d} | train_loss={loss.item():.4f} val_mse={val_mse:.4f} val_rho={rho:.3f}")
        history.append({"epoch": epoch, "train_loss": float(loss.item()),
                        "val_mse": float(val_mse), "val_rho": float(rho)})

        # Keep the best model by val_mse. (Experiment: the val_rho peak overfits;
        # the val_mse minimum generalises better to the test set.)
        if val_mse < best_val - 1e-4:
            best_val = val_mse
            best_rho = rho
            patience_left = patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("[train] early stop")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    model.eval()
    cold_metrics = None
    with torch.no_grad():
        mp_data = _build_message_passing_data(data, train_idx, edge_dropout=0.0)
        emb = model.encode(mp_data)["player"]
        preds_test = model.predict_edge(emb, pair_src[test_idx], pair_dst[test_idx])
        test_mse = torch.nn.functional.mse_loss(preds_test, y[test_idx]).item()
        rho_t = spearmanr(preds_test.cpu().numpy(), y[test_idx].cpu().numpy()).correlation if len(test_idx) > 5 else 0.0

        # Report cold-start separately: test edges whose two players are both unseen
        cold_mask = _cold_start_mask(pair_src.cpu(), pair_dst.cpu(),
                                     train_idx.cpu(), test_idx.cpu()).to(device)
        n_cold = int(cold_mask.sum())
        if n_cold > 5:
            preds_cold = preds_test[cold_mask]
            y_cold = y[test_idx][cold_mask]
            cold_mse = torch.nn.functional.mse_loss(preds_cold, y_cold).item()
            cold_rho = spearmanr(preds_cold.cpu().numpy(), y_cold.cpu().numpy()).correlation
            cold_metrics = {"n_edges": n_cold, "mse": cold_mse, "spearman": cold_rho}
            print(f"[cold-start] n={n_cold} mse={cold_mse:.4f} rho={cold_rho:.3f}")
        else:
            print(f"[cold-start] n={n_cold} — too few, skipped")

    out_pt = MODELS_DIR / f"synergy_hgt{suffix}_{season}.pt"
    torch.save({"state_dict": model.state_dict(),
                "hidden": hidden, "embed_dim": embed_dim,
                "test_mse": test_mse, "test_rho": rho_t}, out_pt)
    (MODELS_DIR / f"metrics{suffix}_{season}.json").write_text(json.dumps({
        "target": target,
        "test_mse": test_mse, "test_spearman": rho_t, "best_val_mse": best_val,
        "best_val_rho": best_rho,
        "split": split, "split_info": split_info,
        "edge_dropout": edge_dropout, "weighted": weighted, "rank_weight": rank_weight,
        "weight_decay": weight_decay, "cold_start": cold_metrics,
        "history": history,
    }, indent=2))
    print(f"[train] saved {out_pt} | test_mse={test_mse:.4f} rho={rho_t:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--season", default="2024-2025")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--split", choices=["player-disjoint", "random"], default="player-disjoint",
                   help="player-disjoint: real generalisation; random: old (leaky) baseline")
    p.add_argument("--edge-dropout", type=float, default=0.0,
                   help="Share of training edges dropped per epoch in message passing. Hurts on this sparse graph; experimental.")
    p.add_argument("--weighted", action="store_true",
                   help="Weight the MSE by sqrt(minutes_together) (down-weights noisy low-minute pairs)")
    p.add_argument("--rank-weight", type=float, default=0.0,
                   help="Pairwise ranking loss weight (Spearman surrogate). 0 = off")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42, help="model initialisation seed")
    p.add_argument("--target", choices=["uplift", "def"], default="uplift",
                   help="uplift: attacking synergy (G+A); def: defensive synergy (xG conceded together)")
    args = p.parse_args()
    train(args.season, epochs=args.epochs, lr=args.lr, split=args.split,
          edge_dropout=args.edge_dropout, weighted=args.weighted,
          rank_weight=args.rank_weight, weight_decay=args.weight_decay, seed=args.seed,
          target=args.target)
