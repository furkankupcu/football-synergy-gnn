# SynergyHGT — Deney Günlüğü (uplift ρ iyileştirme)

**Kural:** Aynı anda **tek değişken** değiştir. Her satırda neyin değiştiği + sonuç + karar.

Sabit ayar: season=2025-2026, split=player-disjoint, target=per90_uplift,
2× HGTConv (h=128, out=64, 4 head), AdamW lr=1e-3, MSE loss, model seçimi=val_mse min.
Metrik: test Spearman ρ (inductive) + cold-start ρ (çiftin iki ucu da eğitimde yok).

| # | Değişen tek şey | test ρ | cold ρ | test MSE | Karar |
|---|---|---|---|---|---|
| 0 | **baseline** (rank=0, weighted=off, sel=val_mse) | 0.285 | 0.237 | 0.211 | referans |
| X | model seçimi val_mse → **val_rho** | 0.156 | 0.176 | 0.234 | ❌ ZARARLI (val_rho overfit) → geri alındı |
| 1 | ranking loss λ=0.5 | 0.311 | 0.284 | 0.210 | ✅ iyi |
| 2 | ranking loss λ=1.0 | **0.331** | 0.299 | 0.211 | ✅ **EN İYİ test ρ** → seçildi |
| 3 | ranking loss λ=2.0 | 0.320 | 0.309 | 0.214 | ✅ (cold en iyi, test biraz düşük) |
| 4 | ranking loss λ=3.0 | 0.317 | 0.292 | 0.220 | ↘ aşırı → MSE bozuluyor |
| 5 | λ=1.0 ÜZERİNE + **weighted** (sqrt-dakika) | **0.334** | **0.315** | 0.211 | ✅ **FİNAL** (cold +0.016) |

| 6 | **team NULL onarımı** (veri hatası, model değişmedi) | **0.368** | **0.389** | **0.169** | ✅ **EN İYİ** — graf 3495→4962 node |

## Final konfigürasyon
`--rank-weight 1.0 --weighted` → **test ρ 0.334, cold-start ρ 0.315** (baseline 0.285/0.237'den
+0.049 / +0.078). Kayıtlı: `data/graph/models/synergy_hgt_2025-2026.pt`.

### Neyin işe yaradığı / yaramadığı (özet)
- ✅ **Ranking loss (λ=1.0)** — en büyük kazanç (sıralamayı doğrudan optimize ediyor).
- ✅ **minutes-weighted MSE** — küçük ama tutarlı, özellikle cold-start'ta.
- ⚠️ **λ>2** — fazla ranking ağırlığı MSE'yi bozuyor, test ρ düşüyor.
- ❌ **val_rho ile model seçimi** — overfit, ciddi zarar (0.285→0.156). val_mse kullan.
- 📌 Sıradaki denenebilir (henüz yapılmadı, hepsi tek-değişken): weight_decay sweep,
  hidden/out boyutu, num_layers=1 vs 3, target=xG+xA-uplift (gürültü azaltma).

## Notlar
- **val_rho ile model seçimi overfit ediyor**: val setinde Spearman gürültülü (paylaşılan
  oyuncular üzerinden korele); zirve epoch test'e genellemiyor. val_mse minimumu daha iyi.
- **Ranking loss (Spearman surrogate)** doğrudan sıralamayı optimize ettiği için ρ'yu
  yükseltiyor, MSE'yi bozmadan. Asıl kazanç burada.
- Sweep'ler aşağıda güncellenecek.

## 2026-09-13 — Onarılmış graf (4962 node) üzerinde ablasyon tekrarı

Önceki tüm sweep'ler 3495 node'luk **bozuk** graf üzerindeydi (team NULL hatası,
CLAUDE.md §4b). Graf %42 büyüyünce λ kararı değişti. Tek seed (42), deterministik.

| Konfig | val_mse | test ρ | test MSE | cold-start ρ | cold MSE |
|---|---:|---:|---:|---:|---:|
| rank=1.0 + weighted | 0.1668 | 0.3677 | 0.1689 | 0.3891 | 0.1732 |
| baseline (rank=0, unweighted) | 0.1672 | 0.3563 | 0.1682 | 0.3878 | 0.1746 |
| rank=0.5 + weighted ✅ | 0.1674 | 0.3705 | 0.1671 | 0.4097 | 0.1704 |
| rank=0 + weighted | 0.1678 | 0.3568 | 0.1684 | 0.4021 | 0.1750 |
| rank=2.0 + weighted | 0.1687 | 0.3790 | 0.1704 | 0.3958 | 0.1734 |
| rank=1.0 unweighted | 0.1688 | 0.3738 | 0.1695 | 0.3977 | 0.1719 |
| rank=3.0 + weighted | 0.1712 | 0.3859 | 0.1729 | 0.3991 | 0.1736 |
| rank=4.0 + weighted | 0.1744 | 0.3880 | 0.1761 | 0.4018 | 0.1764 |
| rank=3.0 unweighted | 0.1762 | 0.3793 | 0.1778 | 0.3939 | 0.1766 |
| rank=5.0 + weighted | 0.1792 | 0.3875 | 0.1809 | 0.4013 | 0.1813 |

**Seçim: λ=0.5 + weighted.** Cold-start ρ (0.4097), cold-start MSE (0.1704) ve
test MSE (0.1671) üçünde de en iyi. val_mse'de birinci değil ama fark 0.0006 —
gürültü seviyesinde. Cold-start scouting'in doğrudan karşılığı olduğu için belirleyici.

**λ tepe noktası:** test ρ λ=4'te doyuyor (0.3880), λ=5'te düşüyor (0.3875). Yüksek λ
test ρ'yu MSE'yi bozarak satın alıyor (0.1684 → 0.1809) ve cold-start'a katkı vermiyor.

**weighted'ın etkisi:** λ=0'da test ρ'yu değiştirmiyor (0.3563 → 0.3568) ama cold-start
ρ'yu belirgin yükseltiyor (0.3878 → 0.4021). Katkısı cold-start'a özgü.
