"""MODEL.md için figür üretici.

Her model değişikliğinden sonra çalıştır:
    python docs/gen_model_figs.py

Çıktılar:
    docs/model_metrics.png      — faz-bazlı test ρ & MSE
    docs/uplift_distribution.png — uplift hedef değişkeni histogramı
"""
from __future__ import annotations

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DOCS = Path(__file__).parent
ROOT = DOCS.parent
GRAPH = ROOT / "data" / "graph"
MODELS = GRAPH / "models"


def fig_metrics() -> None:
    """Faz-bazlı test ρ ve MSE karşılaştırması."""
    # Manuel kayıt — her faz training'inden sonra elle güncellenir.
    phases = [
        ("Baseline\n(joint, random)",        0.789, 0.155, "Şişkin — leakage"),
        ("Faz 1.1\n(joint, P-disjoint)",     0.736, 0.190, "Gerçek genelleme"),
        ("Faz 1.2\n(uplift, P-disjoint)",    0.334, 0.166, "Saf sinerji sinyali"),
        ("Faz 1.3\n(+ cold-start eval)",     0.325, 0.172, "Cold-start ölçüldü"),
        ("Faz 1.4\n(+ team/league emb)",     0.312, 0.173, "Bias ayrıştırma"),
        ("Faz 1.5\n(rank lambda=1 + weighted)",   0.334, 0.211, "Ranking loss"),
        ("Faz 1.6\n(team NULL onarimi)",     0.368, 0.169, "Graf +42% node"),
        ("Faz 1.7\n(lambda=0.5)",            0.371, 0.167, "Ablasyon tekrari"),
    ]
    labels = [p[0] for p in phases]
    rho    = [p[1] for p in phases]
    mse    = [p[2] for p in phases]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    bar_w = 0.38

    c_rho = "#2563eb"
    c_mse = "#dc2626"

    b1 = ax1.bar(x - bar_w/2, rho, bar_w, label="Test Spearman ρ", color=c_rho, alpha=0.85)
    ax1.set_ylabel("Test Spearman ρ", color=c_rho, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=c_rho)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)

    ax2 = ax1.twinx()
    b2 = ax2.bar(x + bar_w/2, mse, bar_w, label="Test MSE", color=c_mse, alpha=0.85)
    ax2.set_ylabel("Test MSE", color=c_mse, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=c_mse)
    ax2.set_ylim(0, 0.25)

    for rect, v in zip(b1, rho):
        ax1.text(rect.get_x() + rect.get_width()/2, v + 0.015, f"{v:.3f}",
                 ha="center", fontsize=9, color=c_rho, fontweight="bold")
    for rect, v in zip(b2, mse):
        ax2.text(rect.get_x() + rect.get_width()/2, v + 0.005, f"{v:.3f}",
                 ha="center", fontsize=9, color=c_mse, fontweight="bold")

    ax1.set_title("SynergyHGT — Faz Karşılaştırması (sezon 2025-2026)", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out = DOCS / "model_metrics.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"yazıldı: {out}")


def fig_uplift_distribution() -> None:
    """Uplift hedef değişkeninin dağılımı."""
    season = "2025-2026"
    f = GRAPH / f"edges_teammate_{season}.parquet"
    if not f.exists():
        print(f"atlandı (yok): {f}")
        return
    df = pd.read_parquet(f)
    if "per90_uplift" not in df.columns:
        print(f"atlandı (per90_uplift yok): {f}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.hist(df["per90_uplift"], bins=80, color="#16a34a", alpha=0.85, edgecolor="white")
    ax1.axvline(0, color="black", lw=1, linestyle="--", alpha=0.6, label="0 (nötr)")
    ax1.axvline(df["per90_uplift"].mean(), color="#dc2626", lw=1.5,
                label=f"μ={df['per90_uplift'].mean():.3f}")
    ax1.set_xlabel("per90_uplift")
    ax1.set_ylabel("kenar sayısı")
    ax1.set_title("Uplift hedef dağılımı (teammate kenarları)")
    ax1.set_xlim(-1.5, 3.0)
    ax1.legend()
    ax1.grid(alpha=0.25)

    ax2.scatter(df["expected_per90"], df["per90_joint_output"],
                s=4, alpha=0.25, color="#0ea5e9")
    lim = max(df["expected_per90"].max(), df["per90_joint_output"].max())
    ax2.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="y=x")
    ax2.set_xlabel("expected per90 (solo_a + solo_b)")
    ax2.set_ylabel("observed pair per90")
    ax2.set_title("Gözlenen vs Beklenen — uplift'in kaynağı")
    ax2.legend()
    ax2.grid(alpha=0.25)

    fig.suptitle("Uplift hedef değişkeni anatomisi", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = DOCS / "uplift_distribution.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"yazıldı: {out}")


def fig_current_metrics_panel() -> None:
    """En güncel metrics_*.json'dan tek-panel summary kart."""
    metric_files = sorted(MODELS.glob("metrics_*.json"))
    if not metric_files:
        print("atlandı: metrics_*.json yok")
        return
    m = json.loads(metric_files[-1].read_text())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    season = metric_files[-1].stem.replace("metrics_", "")
    cold = m.get("cold_start") or {}
    info = m.get("split_info", {})
    pl   = info.get("players", {})
    eg   = info.get("edges", {})

    rows = [
        ("Sezon",            season),
        ("Split",            m.get("split", "?")),
        ("Edge dropout",     str(m.get("edge_dropout", 0))),
        ("",                 ""),
        ("Players (tr/v/te)", f"{pl.get('train','?')} / {pl.get('val','?')} / {pl.get('test','?')}"),
        ("Edges (tr/v/te)",   f"{eg.get('train','?')} / {eg.get('val','?')} / {eg.get('test','?')}"),
        ("",                 ""),
        ("Test ρ",           f"{m.get('test_spearman', 0):.3f}"),
        ("Test MSE",         f"{m.get('test_mse', 0):.4f}"),
        ("Best val MSE",     f"{m.get('best_val_mse', 0):.4f}"),
        ("",                 ""),
        ("Cold-start n",     str(cold.get("n_edges", "?"))),
        ("Cold-start ρ",     f"{cold.get('spearman', 0):.3f}" if cold else "?"),
        ("Cold-start MSE",   f"{cold.get('mse', 0):.4f}" if cold else "?"),
    ]

    y = 0.95
    for k, v in rows:
        if not k and not v:
            y -= 0.04; continue
        ax.text(0.04, y, k, fontsize=10.5, fontweight="bold", color="#1f2937")
        ax.text(0.55, y, v, fontsize=10.5, color="#111827", family="monospace")
        y -= 0.06

    ax.set_title("Güncel Model Metrikleri", fontsize=13, fontweight="bold", loc="left")
    fig.tight_layout()
    out = DOCS / "current_metrics.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"yazıldı: {out}")


if __name__ == "__main__":
    fig_metrics()
    fig_uplift_distribution()
    fig_current_metrics_panel()
