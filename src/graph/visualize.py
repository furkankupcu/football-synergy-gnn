"""SynergyHGT görselleştirme — mimari, pipeline, metrik, çıktı.

Üretilen PNG'ler docs/ altına yazılır:
  model_architecture.png   — katman katman mimari
  pipeline.png             — scrape → edge_builder → build → train akışı
  ablation_metrics.png     — deney karşılaştırması (test ρ / cold ρ)
  training_curves.png      — epoch boyunca val_mse & val_rho
  pred_vs_actual.png       — test kenarlarında tahmin vs gerçek uplift (+ρ)

Kullanım:
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

BLUE = "#2c6fbb"; GREEN = "#2e8b57"; ORANGE = "#e07b39"; GREY = "#5a5a5a"; RED = "#c0392b"


# --------------------------------------------------------------------------- 1) mimari
def fig_architecture(path: Path = DOCS / "model_architecture.png") -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 14); ax.axis("off")
    ax.set_title("SynergyHGT — Mimari", fontsize=15, fontweight="bold")

    def box(x, y, w, h, text, color, fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=color, ec="black", alpha=0.85, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=16, lw=1.4, color=GREY))

    box(0.4, 12.4, 4.2, 1.2, "Node feature (47-dim)\nper90 stat·z + age/height\n+ foot(3) + pozisyon(11)", BLUE, 8.5)
    box(5.4, 12.4, 4.2, 1.2, "team_emb (8) + league_emb (4)\nöğrenilebilir embedding", GREEN, 8.5)
    box(2.4, 10.8, 5.2, 0.9, "concat → player.x (59-dim)", GREY)
    box(2.4, 9.5, 5.2, 0.9, "lin_in: Linear(→128)  [node-type başına]", BLUE)
    box(2.4, 8.2, 5.2, 0.9, "HGTConv #1  128→128, 4 head → GELU", ORANGE)
    box(2.4, 6.9, 5.2, 0.9, "HGTConv #2  128→128, 4 head → GELU", ORANGE)
    box(2.4, 5.6, 5.2, 0.9, "lin_out: Linear(128→64)  → embedding", BLUE)
    box(2.4, 4.1, 5.2, 1.0, "Edge regressor girişi (çift a,b):\n[eₐ | e_b | |eₐ−e_b| | eₐ·e_b]  (256)", GREEN)
    box(2.4, 2.6, 5.2, 0.9, "MLP: 256→128→64→1", BLUE)
    box(3.1, 1.2, 3.8, 0.9, "per90_uplift tahmini", RED)

    for y1, y2 in [(12.4, 11.7), (11.7, 10.4), (9.5, 9.1)]:
        pass
    arrow(2.5, 12.4, 4.0, 11.7); arrow(7.5, 12.4, 6.0, 11.7)
    arrow(5.0, 10.8, 5.0, 10.4); arrow(5.0, 9.5, 5.0, 9.1)
    arrow(5.0, 8.2, 5.0, 7.8); arrow(5.0, 6.9, 5.0, 6.5)
    arrow(5.0, 5.6, 5.0, 5.1); arrow(5.0, 4.1, 5.0, 3.5)
    arrow(5.0, 2.6, 5.0, 2.1)

    ax.text(9.85, 7.3, "mesaj geçişi:\nteammate + opponent\nkenarları (HGT attention\nher ilişkiye ayrı)",
            ha="right", va="center", fontsize=8, color=ORANGE, style="italic")
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {path}")


# --------------------------------------------------------------------------- 2) pipeline
def fig_pipeline(path: Path = DOCS / "pipeline.png") -> None:
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    ax.set_title("Veri → GNN Pipeline", fontsize=14, fontweight="bold")
    steps = [
        ("Maç verisi\n(9 lig)", BLUE, "matches, player_stats,\nshots, avg_positions"),
        ("edge_builder", ORANGE, "teammate 37.7k\nopponent 117k\n+ uplift label"),
        ("build", GREEN, "HeteroData\n3730 düğüm\n47 feature"),
        ("train\nSynergyHGT", RED, "test ρ, cold ρ\nmodel .pt"),
    ]
    w = 2.2; gap = 0.85; x = 0.4
    for i, (name, color, sub) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((x, 1.1), w, 1.1, boxstyle="round,pad=0.08",
                                    fc=color, ec="black", alpha=0.85, lw=1.2))
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
    ax.set_ylabel("Spearman ρ"); ax.set_title("Ablasyon — uplift ρ (tek değişken)", fontweight="bold")
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
        print("history yok, atlandı"); return
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
    ax1.text(ep[best], ax1.get_ylim()[1]*0.97, f" seçilen epoch {ep[best]}",
             color=RED, fontsize=8, va="top")
    ax1.set_title(f"Eğitim eğrileri (rank_weight={m.get('rank_weight')}, "
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

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, pred, s=8, alpha=0.25, color=BLUE)
    lim = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    ax.plot(lim, lim, ls="--", color=RED, lw=1)
    ax.set_xlabel("gerçek per90_uplift"); ax.set_ylabel("tahmin")
    ax.set_title(f"Test kenarları: tahmin vs gerçek\nSpearman ρ = {rho:.3f}  (n={len(pred)})",
                 fontweight="bold")
    ax.grid(alpha=0.3)
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
