# Opposition Scout Report — [TEAM NAME]

> **Şablon kullanımı:** Bu dosyayı `reports/opposition/[team_slug].md` altına kopyala, başlıkları doldur. EN başlık + TR/EN karışık gövde stratejisi: uluslararası okuyucu özet okuyabilsin, Türk okuyucu derinleşsin.
> 
> Kaynak referansları: [PFSA Opposition Template](https://thepfsa.co.uk/wp-content/uploads/2025/08/PFSA_Scout-Report_Opposition-Report-Master-Copy.pdf), [Schalke 04 Opposition Analysis](https://www.slideshare.net/petrchulkov/schalke-04feb), [Chris Gill — Opposition Report essentials](https://www.linkedin.com/pulse/what-include-opposition-report-chris-gill).

---

## 0. Meta

| | |
|---|---|
| **Team / Takım** | |
| **Coach / Teknik Direktör** | |
| **Competition / Turnuva** | WC 2026 — Group [X] / [Knockout Round] |
| **Report date / Rapor tarihi** | YYYY-MM-DD |
| **Matches analyzed / İncelenen maçlar** | 3 most recent (date, opponent, result, venue) |
| **Analyst / Analist** | Furkan Küpçü |
| **Version / Versiyon** | v0.1 |

---

## 1. Executive Summary — 3 cümle

**EN (mandatory):**
- *Identity:* [team] are a [possession/transition/defensive-block] side, typically setting up in a [formation] under [coach].
- *Key strength:* [one sentence — e.g. "elite transition speed through wide overloads"].
- *Key weakness vs Türkiye:* [one sentence — e.g. "high defensive line vulnerable to direct play behind"].

**TR (opsiyonel ek not):** Türkiye için kilit tehdit ve kilit fırsat tek cümleyle.

---

## 2. Formation & First XI

**Base formation:** [4-3-3 / 4-2-3-1 / 3-5-2 / etc.]
**Predicted XI vs Türkiye:** _(based on last 3 matches + injury news)_

```
                [GK]
        [LB] [LCB] [RCB] [RB]
            [CDM]  [CM]
       [LW]  [CAM]  [RW]
                [ST]
```

| Pos | Player | Age | Club | Key trait |
|---|---|---|---|---|
| GK | | | | |
| RB | | | | |
| RCB | | | | |
| LCB | | | | |
| LB | | | | |
| CDM | | | | |
| CM | | | | |
| CAM | | | | |
| RW | | | | |
| LW | | | | |
| ST | | | | |

**Bench / Probable subs:** 5–7 isim, gelişe göre değişebilen kart.

---

## 3. In-Possession Patterns — Hücum

### 3.1 Build-up
- **Goalkeeper involvement:** [short/long, side preference]
- **First line out:** [centre-backs split, FB high, DM drops, etc.]
- **Provocation:** Rakip pres altında [risk alır / uzun atar / yatay rotasyon]
- **Key data:** PPDA [x], passes per defensive action — qualifying matches average
- **Reference clips:** [match-date min'sec], [match-date min'sec]

### 3.2 Progression
- **Through midfield:** [number 10 dropping, fullback inverted, CB carry, etc.]
- **Wide vs central preference:** [%]
- **Switch of play frequency:** [high/medium/low]

### 3.3 Final third
- **Crossing volume + zones:** [byline / cutback / early cross]
- **Cut-in patterns:** [inverted wingers, RW-CAM combinations]
- **Box presence:** Average bodies in box on cross [x]

---

## 4. Out-of-Possession — Defans

### 4.1 Pressing
- **High press trigger:** [back pass / wide trap / GK touches ball]
- **Press structure:** [aggressive 4-4-2 / passive mid-block / hybrid]
- **PPDA when pressing high:** [x]

### 4.2 Mid/low block
- **Defensive line height:** [yards from goal]
- **Compactness:** [vertical compaction during attack vs out]
- **Marking scheme:** [zonal / man / hybrid in box]

### 4.3 Transition (rakibin defense → opp. attack)
- **Counter-press intensity:** [seconds to first defensive action after losing ball]
- **Vulnerability zones:** _(en sıkça gol yedikleri bölge)_
  - Right channel / left channel / between CBs / behind FB

---

## 5. Set Pieces

### 5.1 Attacking
- **Corners:** [near post / far post / short / late runner]
- **Free kicks:** [direct shooters: PLAYER, in-swing/out-swing]
- **Throw-ins:** [long throw? specialist?]

### 5.2 Defending
- **Corner defense:** [zonal / man / mixed]
- **Free kick wall:** [size, position]
- **Aerial vulnerability:** Headers conceded per 90 [x]

---

## 6. Key Players — derinlemesine 3 isim

### 6.1 [Player 1 — usually the talisman]
- Heat map zone, dominant foot, key strength, key weakness, Türkiye için bağ.

### 6.2 [Player 2]
- Aynı format.

### 6.3 [Player 3]
- Aynı format.

---

## 7. Türkiye İçin Analiz

### 7.1 Threats — savunma riskleri
1. **[Threat 1]** — örn. "fullback'lerimiz vs hızlı kanat oyuncuları, isolations"
2. **[Threat 2]** — örn. "set-piece kaybedildiğinde hızlı transition"
3. **[Threat 3]**

### 7.2 Opportunities — hücum yolları
1. **[Opportunity 1]** — örn. "yüksek defansif çizgi → Akgün/Akturkoğlu derinliği"
2. **[Opportunity 2]**
3. **[Opportunity 3]**

### 7.3 GNN Sinerji Önerisi _(repodaki SynergyHGT model çıktısı)_

Türkiye'nin first XI'i için bu maça özel **en yüksek sinerji** kombinasyonları:

| Çift | Sinerji skoru | Pozisyon | Sezon kanıtı |
|---|---|---|---|
| | | | |
| | | | |

**Not:** Skor `model.predict_edge` çıktısıdır; sezon içi kulüp performansı baz alınır. Milli takım kompozisyonu (farklı liglerden oyuncular) için yaklaşıktır.

---

## 8. Tactical Recommendation — kısa

**Türkiye için 3 noktalı plan:**
1. **Formasyon:** Önerilen sistem ve neden.
2. **Tetikleyici an:** Maçın hangi anında ne yapılmalı (örn. "rakip pres'i tetiklediğinde, kanat oyuncusu yatık koşu").
3. **Substitusyon planı:** 60-75. dakikalar.

---

## 9. Visual Annex — eklenecek görseller

- [ ] Predicted XI on pitch (mplsoccer)
- [ ] Average position heatmap (3 maç ortalama)
- [ ] xG/xGA trend grafiği (eleme/hazırlık maçları)
- [ ] Set-piece freeze frame (2 örnek)
- [ ] Türkiye için GNN sinerji haritası (mplsoccer + line)

---

## 10. Sources & Methodology

- **Match footage:** [maç tarihleri ve kaynak: FIFA+, FOOTreplays, official YouTube]
- **Data:** [FBref, FotMob, StatsBomb if any]
- **Tactical analysis read:** [The Athletic, Spielverlagerung, Coaches' Voice — varsa link]
- **Personal observation hours:** [tahmini izleme süresi]

---

## Changelog

| Date | Version | Note |
|---|---|---|
| YYYY-MM-DD | v0.1 | Initial draft |

---

> **Kullanım sonrası:** Bu raporu PDF olarak export et (`pandoc opposition_report.md -o reports/opposition/[team]_v1.pdf`), Medium'da kısaltılmış versiyon yayınla, LinkedIn'de carousel olarak görselleri paylaş.
