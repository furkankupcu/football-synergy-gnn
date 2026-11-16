"""SynergyHGT figures — architecture, pipeline, metrics, predictions.

PNGs are written to docs/:
  model_architecture.png   — layer-by-layer architecture
  pipeline.png             — data → edge_builder → build → train
  ablation_metrics.png     — single-variable ablation (test ρ / cold-start ρ)
  training_curves.png      — val_mse & val_rho per epoch
  pred_vs_actual.png       — predicted vs actual uplift on test edges

Usage:
  python -m src.graph.visualize --season 2025-2026
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

DOCS = Path("docs")
DOCS.mkdir(exist_ok=True)
GRAPH_DIR = Path("data/graph")
MODELS_DIR = GRAPH_DIR / "models"

BLUE = "#2563eb"; GREEN = "#059669"; ORANGE = "#ea580c"; GREY = "#64748b"; RED = "#dc2626"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlepad": 12,
    "axes.edgecolor": "#94a3b8", "axes.labelcolor": "#334155",
    "xtick.color": "#475569", "ytick.color": "#475569", "legend.frameon": False,
})


# --------------------------------------------------------------------------- 1) mimari
def fig_architecture(path: Path = DOCS / "model_architecture.png") -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 14); ax.axis("off")
    ax.set_title("SynergyHGT — architecture", fontsize=15, fontweight="bold")

    def box(x, y, w, h, text, color, fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=color, ec="none", alpha=0.92))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=16, lw=1.4, color=GREY))

    box(0.4, 12.4, 4.2, 1.2, "Node features (47-dim)\nper-90 stats (z) + age/height\n+ foot (3) + position one-hot (13)", BLUE, 8.5)
    box(5.4, 12.4, 4.2, 1.2, "team_emb (8) + league_emb (4)\nlearned embeddings", GREEN, 8.5)
    box(2.4, 10.8, 5.2, 0.9, "concat → player.x (59-dim)", GREY)
    box(2.4, 9.5, 5.2, 0.9, "lin_in: Linear(→128)  [per node type]", BLUE)
    box(2.4, 8.2, 5.2, 0.9, "HGTConv #1  128→128, 4 heads → GELU", ORANGE)
    box(2.4, 6.9, 5.2, 0.9, "HGTConv #2  128→128, 4 heads → GELU", ORANGE)
    box(2.4, 5.6, 5.2, 0.9, "lin_out: Linear(128→64)  → player embedding", BLUE)
    box(2.4, 4.1, 5.2, 1.0, "Edge regressor input (pair a, b):\n[eₐ | e_b | |eₐ−e_b| | eₐ·e_b]  (256)", GREEN)
    box(2.4, 2.6, 5.2, 0.9, "MLP: 256→128→64→1", BLUE)
    box(3.1, 1.2, 3.8, 0.9, "predicted per90_uplift", RED)

    arrow(2.5, 12.4, 4.0, 11.7); arrow(7.5, 12.4, 6.0, 11.7)
    arrow(5.0, 10.8, 5.0, 10.4); arrow(5.0, 9.5, 5.0, 9.1)
    arrow(5.0, 8.2, 5.0, 7.8); arrow(5.0, 6.9, 5.0, 6.5)
    arrow(5.0, 5.6, 5.0, 5.1); arrow(5.0, 4.1, 5.0, 3.5)
    arrow(5.0, 2.6, 5.0, 2.1)

    ax.text(9.85, 7.3, "message passing over\nteammate + opponent edges\n(separate HGT attention\nper relation)",
            ha="right", va="center", fontsize=8, color=ORANGE, style="italic")
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {path}")


# --------------------------------------------------------------------------- 2) pipeline
def fig_pipeline(season: str = "2025-2026", path: Path = DOCS / "pipeline.png") -> None:
    # Sayılar elle yazılınca bayatlıyordu → mevcut graftan oku
    n_nodes = n_tm = n_op = n_feat = "?"
    pt = GRAPH_DIR / f"hetero_{season}.pt"
    if pt.exists():
        import torch
        d = torch.load(pt, weights_only=False)
        n_nodes, n_feat = d["player"].x.shape
        n_tm = f"{d['player', 'teammate', 'player'].edge_index.shape[1] // 2 / 1000:.1f}k"
        n_op = f"{d['player', 'opponent', 'player'].edge_index.shape[1] // 2 / 1000:.0f}k"
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    ax.set_title("Data → GNN pipeline", fontsize=14, fontweight="bold")
    steps = [
        ("Match data\n(9 leagues)", BLUE, "matches, player_stats,\navg_positions"),
        ("edge_builder", ORANGE, f"teammate {n_tm}\nopponent {n_op}\n+ uplift label"),
        ("build", GREEN, f"HeteroData\n{n_nodes:,} players\n{n_feat} features"),
        ("train\nSynergyHGT", RED, "test ρ, cold-start ρ\nmodel .pt"),
    ]
    w = 2.2; gap = 0.85; x = 0.4
    for i, (name, color, sub) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((x, 1.1), w, 1.1, boxstyle="round,pad=0.08",
                                    fc=color, ec="none", alpha=0.92))
        ax.text(x + w / 2, 1.65, name, ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")
        ax.text(x + w / 2, 0.55, sub, ha="center", va="center", fontsize=7.5, color=GREY)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w, 1.65), (x + w + gap, 1.65),
                         arrowstyle="-|>", mutation_scale=18, lw=1.6, color=GREY))
        x += w + gap
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {path}")


# --------------------------------------------------------------------------- 3) ablation
def fig_ablation(results: list[dict], path: Path = DOCS / "ablation_metrics.png") -> None:
    labels = [r["label"] for r in results]
    test_rho = [r["test_rho"] for r in results]
    cold_rho = [r["cold_rho"] for r in results]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.5), 4.5))
    b1 = ax.bar(x - w/2, test_rho, w, label="test ρ (inductive)", color=BLUE)
    b2 = ax.bar(x + w/2, cold_rho, w, label="cold-start ρ", color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Spearman ρ"); ax.set_title("Ablation — uplift ρ (one change at a time)", fontweight="bold")
    ax.axhline(results[0]["test_rho"], ls="--", color=GREY, lw=1, alpha=0.7)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f"{b.get_height():.3f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {path}")


# --------------------------------------------------------------------------- 4) eğitim eğrileri
def fig_training_curves(season: str, path: Path = DOCS / "training_curves.png") -> None:
    mfile = MODELS_DIR / f"metrics_{season}.json"
    m = json.loads(mfile.read_text())
    hist = m.get("history", [])
    if not hist:
        print("no history, skipped"); return
    ep = [h["epoch"] for h in hist]
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.plot(ep, [h["val_mse"] for h in hist], color=BLUE, label="val_mse")
    ax1.plot(ep, [h["train_loss"] for h in hist], color=GREY, ls="--", alpha=0.7, label="train_loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss / MSE", color=BLUE)
    ax2 = ax1.twinx()
    ax2.plot(ep, [h["val_rho"] for h in hist], color=GREEN, label="val_rho")
    ax2.set_ylabel("val Spearman ρ", color=GREEN)
    best = int(np.argmin([h["val_mse"] for h in hist]))
    ax1.axvline(ep[best], color=RED, ls=":", lw=1.3)
    ax1.text(ep[best], ax1.get_ylim()[1]*0.97, f" selected epoch {ep[best]} (min val_mse)",
             color=RED, fontsize=8, va="top")
    ax1.set_title(f"Training curves (rank_weight={m.get('rank_weight')}, "
                  f"weighted={m.get('weighted')})", fontweight="bold")
    l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {path}")


# --------------------------------------------------------------------------- 5) pred vs actual
def fig_pred_scatter(season: str, path: Path = DOCS / "pred_vs_actual.png") -> None:
    import torch
    from scipy.stats import spearmanr
    from src.graph.model import SynergyHGT
    from src.graph.train import _split_edges_player_disjoint, _build_message_passing_data

    data = torch.load(GRAPH_DIR / f"hetero_{season}.pt", weights_only=False)
    et = ("player", "teammate", "player")
    ei = data[et].edge_index; half = ei.shape[1] // 2
    ps, pd_ = ei[0, :half], ei[1, :half]; y = data[et].y[:half]
    tr_idx, _, te_idx, _ = _split_edges_player_disjoint(ps, pd_, data["player"].x.shape[0])

    ckpt = torch.load(MODELS_DIR / f"synergy_hgt_{season}.pt", weights_only=False)
    model = SynergyHGT(metadata=data.metadata(), in_dim=data["player"].x.shape[1],
                       hidden=ckpt["hidden"], out=ckpt["embed_dim"],
                       num_teams=int(getattr(data["player"], "num_teams", 0)) or None,
                       num_leagues=int(getattr(data["player"], "num_leagues", 0)) or None)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    with torch.no_grad():
        mp = _build_message_passing_data(data, tr_idx, edge_dropout=0.0)
        emb = model.encode(mp)["player"]
        pred = model.predict_edge(emb, ps[te_idx], pd_[te_idx]).numpy()
    actual = y[te_idx].numpy()
    rho = spearmanr(pred, actual).correlation

    fig, ax = plt.subplots(figsize=(6.4, 6))
    # Binlerce üst üste nokta → yoğunluk (hexbin) okunaklı; uç değerler eksen dışı kalsın
    lo, hi = np.percentile(np.r_[actual, pred], [0.5, 99.5])
    hb = ax.hexbin(actual, pred, gridsize=45, extent=(lo, hi, lo, hi), mincnt=1,
                   cmap="Blues", bins="log", linewidths=0)
    ax.plot([lo, hi], [lo, hi], ls="--", color=RED, lw=1, label="perfect prediction")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("actual per90_uplift"); ax.set_ylabel("predicted per90_uplift")
    ax.set_title(f"Test edges (unseen players)\nSpearman ρ = {rho:.3f}  ·  n = {len(pred):,}")
    fig.colorbar(hb, ax=ax, label="edges (log)", shrink=0.8)
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2025-2026")
    ap.add_argument("--ablation-json", default=None,
                    help="Ablasyon sonuçları JSON (list of {label,test_rho,cold_rho})")
    a = ap.parse_args()

    fig_architecture()
    fig_pipeline()
    fig_training_curves(a.season)
    fig_pred_scatter(a.season)
    if a.ablation_json and Path(a.ablation_json).exists():
        fig_ablation(json.loads(Path(a.ablation_json).read_text()))


if __name__ == "__main__":
    main()
