# SynergyHGT — Experiment log (uplift ρ)

**Rule:** change **one variable at a time**. Each row records what changed, the result and the decision.

Fixed setup: season 2025-2026, player-disjoint split, target `per90_uplift`,
2× HGTConv (hidden 128, out 64, 4 heads), AdamW lr 1e-3, model selection = min val_mse.
Metrics: test Spearman ρ (inductive) and cold-start ρ (both players of the pair unseen in training).

## Round 1 — graph with 3,495 players (before the missing-team data fix)

| # | Single change | test ρ | cold ρ | test MSE | Decision |
|---|---|---:|---:|---:|---|
| 0 | **baseline** (rank=0, unweighted, select on val_mse) | 0.285 | 0.237 | 0.211 | reference |
| X | model selection val_mse → **val_rho** | 0.156 | 0.176 | 0.234 | ❌ harmful (val_rho overfits) → reverted |
| 1 | ranking loss λ=0.5 | 0.311 | 0.284 | 0.210 | ✅ |
| 2 | ranking loss λ=1.0 | **0.331** | 0.299 | 0.211 | ✅ best test ρ → kept |
| 3 | ranking loss λ=2.0 | 0.320 | 0.309 | 0.214 | best cold-start, lower test ρ |
| 4 | ranking loss λ=3.0 | 0.317 | 0.292 | 0.220 | ↘ too strong, MSE degrades |
| 5 | λ=1.0 **+ minutes-weighted MSE** | **0.334** | **0.315** | 0.211 | ✅ final for round 1 |
| 6 | **data fix: missing team ids** (model unchanged) | **0.368** | **0.389** | **0.169** | ✅ graph grew 3,495 → 4,962 players |

What worked / what did not:
- ✅ **Ranking loss** (Spearman surrogate) — the largest gain: it optimises the ordering directly.
- ✅ **Minutes-weighted MSE** (√minutes together) — small but consistent, mostly on cold-start.
- ⚠️ **λ > 2** — buys test ρ by hurting MSE.
- ❌ **Selecting the checkpoint on val ρ** — overfits badly (0.285 → 0.156). Spearman on the
  validation set is noisy (edges share players); the val_mse minimum generalises better.

## Round 2 — repeated on the repaired graph (4,962 players), 2026-09-13

All earlier sweeps ran on the broken graph, so the λ decision was re-tested. Single seed (42), deterministic.

| Config | val_mse | test ρ | test MSE | cold-start ρ | cold MSE |
|---|---:|---:|---:|---:|---:|
| rank=1.0 + weighted | 0.1668 | 0.3677 | 0.1689 | 0.3891 | 0.1732 |
| baseline (rank=0, unweighted) | 0.1672 | 0.3563 | 0.1682 | 0.3878 | 0.1746 |
| **rank=0.5 + weighted ✅** | 0.1674 | 0.3705 | 0.1671 | 0.4097 | 0.1704 |
| rank=0 + weighted | 0.1678 | 0.3568 | 0.1684 | 0.4021 | 0.1750 |
| rank=2.0 + weighted | 0.1687 | 0.3790 | 0.1704 | 0.3958 | 0.1734 |
| rank=1.0 unweighted | 0.1688 | 0.3738 | 0.1695 | 0.3977 | 0.1719 |
| rank=3.0 + weighted | 0.1712 | 0.3859 | 0.1729 | 0.3991 | 0.1736 |
| rank=4.0 + weighted | 0.1744 | 0.3880 | 0.1761 | 0.4018 | 0.1764 |
| rank=3.0 unweighted | 0.1762 | 0.3793 | 0.1778 | 0.3939 | 0.1766 |
| rank=5.0 + weighted | 0.1792 | 0.3875 | 0.1809 | 0.4013 | 0.1813 |

**Choice: λ=0.5 + weighted.** Best on cold-start ρ, cold-start MSE and test MSE. It is not first
on val_mse, but the gap (0.0006) is at noise level. Cold-start is what scouting a new signing
actually looks like, so it decides.

**λ saturates:** test ρ peaks at λ=4 (0.3880) and drops at λ=5. Higher λ buys test ρ by
degrading MSE (0.1684 → 0.1809) and does not help cold-start.

**Effect of weighting:** at λ=0 it leaves test ρ unchanged (0.3563 → 0.3568) but clearly lifts
cold-start ρ (0.3878 → 0.4021).

Configurations were **not** chosen on test ρ: picking the best of ten on the test set would
turn it into a validation set.

## Next single-variable experiments (not run yet)

weight-decay sweep · hidden/output size · 1 vs 3 HGT layers · xG+xA-based uplift (less noisy than goals+assists).
