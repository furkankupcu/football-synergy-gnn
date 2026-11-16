# Opposition Scout Report — [TEAM NAME]

> **Template usage:** copy this file to `reports/opposition/[team_slug].md` and fill in the sections. Short summaries come first so an international reader can skim; the detailed sections go deeper.
> 
> References: [PFSA Opposition Template](https://thepfsa.co.uk/wp-content/uploads/2025/08/PFSA_Scout-Report_Opposition-Report-Master-Copy.pdf), [Schalke 04 Opposition Analysis](https://www.slideshare.net/petrchulkov/schalke-04feb), [Chris Gill — Opposition Report essentials](https://www.linkedin.com/pulse/what-include-opposition-report-chris-gill).

---

## 0. Meta

| | |
|---|---|
| **Team** | |
| **Coach** | |
| **Competition** | WC 2026 — Group [X] / [Knockout Round] |
| **Report date** | YYYY-MM-DD |
| **Matches analyzed** | 3 most recent (date, opponent, result, venue) |
| **Analyst** | Furkan Küpçü |
| **Version** | v0.1 |

---

## 1. Executive Summary — 3 sentences

- *Identity:* [team] are a [possession/transition/defensive-block] side, typically setting up in a [formation] under [coach].
- *Key strength:* [one sentence — e.g. "elite transition speed through wide overloads"].
- *Key weakness vs Türkiye:* [one sentence — e.g. "high defensive line vulnerable to direct play behind"].

**Optional note:** the key threat and the key opportunity for Türkiye in one sentence.

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

**Bench / Probable subs:** 5–7 names; may change with late news.

---

## 3. In-Possession Patterns

### 3.1 Build-up
- **Goalkeeper involvement:** [short/long, side preference]
- **First line out:** [centre-backs split, FB high, DM drops, etc.]
- **Under pressure:** when pressed they [take risks / go long / rotate sideways]
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

## 4. Out-of-Possession

### 4.1 Pressing
- **High press trigger:** [back pass / wide trap / GK touches ball]
- **Press structure:** [aggressive 4-4-2 / passive mid-block / hybrid]
- **PPDA when pressing high:** [x]

### 4.2 Mid/low block
- **Defensive line height:** [yards from goal]
- **Compactness:** [vertical compaction during attack vs out]
- **Marking scheme:** [zonal / man / hybrid in box]

### 4.3 Transition (their defence → opponent attack)
- **Counter-press intensity:** [seconds to first defensive action after losing ball]
- **Vulnerability zones:** _(where they concede most often)_
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

## 6. Key Players — 3 in depth

### 6.1 [Player 1 — usually the talisman]
- Heat map zone, dominant foot, key strength, key weakness, relevance for Türkiye.

### 6.2 [Player 2]
- Same format.

### 6.3 [Player 3]
- Same format.

---

## 7. Analysis for Türkiye

### 7.1 Threats — defensive risks
1. **[Threat 1]** — e.g. "our fullbacks vs fast wingers, isolations"
2. **[Threat 2]** — e.g. "quick transition after losing a set piece"
3. **[Threat 3]**

### 7.2 Opportunities — attacking routes
1. **[Opportunity 1]** — e.g. "high defensive line → depth runs from Akgün/Aktürkoğlu"
2. **[Opportunity 2]**
3. **[Opportunity 3]**

### 7.3 GNN Synergy Suggestion _(SynergyHGT model output from this repo)_

The **highest-synergy** combinations in Türkiye's first XI for this match:

| Pair | Synergy score | Position | Season evidence |
|---|---|---|---|
| | | | |
| | | | |

**Note:** the score is the `model.predict_edge` output and is based on in-season club performance. For a national team (players from different leagues) it is an approximation.

---

## 8. Tactical Recommendation — short

**3-point plan for Türkiye:**
1. **Formation:** the recommended system and why.
2. **Trigger moment:** what to do at which point of the match (e.g. "when the opponent triggers their press, the winger makes a diagonal run").
3. **Substitution plan:** minutes 60-75.

---

## 9. Visual Annex — visuals to add

- [ ] Predicted XI on pitch (mplsoccer)
- [ ] Average position heatmap (3-match average)
- [ ] xG/xGA trend chart (qualifiers/friendlies)
- [ ] Set-piece freeze frame (2 examples)
- [ ] GNN synergy map for Türkiye (mplsoccer + lines)

---

## 10. Sources & Methodology

- **Match footage:** [match dates and source: FIFA+, FOOTreplays, official YouTube]
- **Data:** [FBref, FotMob, StatsBomb if any]
- **Tactical analysis read:** [The Athletic, Spielverlagerung, Coaches' Voice — links if any]
- **Personal observation hours:** [estimated viewing time]

---

## Changelog

| Date | Version | Note |
|---|---|---|
| YYYY-MM-DD | v0.1 | Initial draft |

---

> **After use:** export this report as PDF (`pandoc opposition_report.md -o reports/opposition/[team]_v1.pdf`), publish a shorter version on Medium and share the visuals as a LinkedIn carousel.
