# Position Evaluation Metrics

Player evaluation metrics by position.
column names are given in parentheses.

---

## Center Backs

### Ball-Playing CB
**What they do:** Initiate build-up from the back, break opposition lines with passes, and maintain composure under pressure.

| Metric | Source Column | Notes |
|--------|-----------------|-------|
| Passes per 90 | `total_pass` | High volume suggests trusted ball progressor |
| Progressive passes per 90 | `total_progression` | Ball progression ability |
| Accurate progressive passes % | `total_progression` / `total_pass` | Ball progression ability |
| Accurate long passes % | `accurate_long_balls` / `total_long_balls` | Passing range |
| Interceptions per 90 | `interception_won` | Reading play early |

---

### Combative CB / Stopper
**What they do:** Win duels, clear danger, mark target forwards, and contest aerial challenges.

| Metric | Source Column | Notes |
|--------|-----------------|-------|
| Defensive duels per 90 | `duel_won` + `duel_lost` | High volume to fit profile |
| Defensive duels won % | `duel_won` / (`duel_won` + `duel_lost`) | Success rate points to ability |
| Aerial duels per 90 | `aerial_won` + `aerial_lost` | Volume and ability in aerial duels |
| Won aerial duels % | `aerial_won` / (`aerial_won` + `aerial_lost`) | Similarly, volume and ability |
| Shots blocked per 90 | `outfielder_block` | Defence of the penalty area |

---

### Libero / Middle CB
**What they do:** Sweep behind the backline, support both build-up and cover, dictate transitions from deep.

| Metric | Source Column | Notes |
|--------|-----------------|-------|
| Defensive duels per 90 | `duel_won` + `duel_lost` | Below-average volume to fit less confrontational profile |
| Defensive duels won % | `duel_won` / (`duel_won` + `duel_lost`) | Good success rate when called into action |
| Interceptions per 90 | `interception_won` | Reading play early |
| Accurate passes % | `accurate_pass` / `total_pass` | Reliable in possession |
| Accurate long passes % | `accurate_long_balls` / `total_long_balls` | Passing range |

---

### Pin CB / Wide CB (in 3)
**What they do:** Cover half-spaces, support full backs, defend wide-to-central transitions, and build down their respective flank.

| Metric | Source Column | Notes |
|--------|-----------------|-------|
| Defensive duels per 90 | `duel_won` + `duel_lost` | High volume to fit profile |
| Defensive duels won % | `duel_won` / (`duel_won` + `duel_lost`) | Success rate points to ability |
| Interceptions per 90 | `interception_won` | Reading play early |
| Progressive passes per 90 | `total_progression` | Ability to step forward into midfield and distribute |
| Accurate progressive passes % | `total_progression` / `total_pass` | Ability to step forward into midfield and distribute |

---

## Midfielders

### Defensive Midfielder (#6)
**What they do:** Screen the backline, intercept passes, dictate tempo from deep, recycle possession, and break up opposition transitions.

| Metric | Source Column | Notes |
|--------|-----------------|-------|
| Interceptions per 90 | `interception_won` | Reading play early |
| Defensive duels per 90 | `duel_won` + `duel_lost` | High volume to fit profile |
| Defensive duels won % | `duel_won` / (`duel_won` + `duel_lost`) | Success rate points to ability |
| Accurate passes % | `accurate_pass` / `total_pass` | Reliable in possession |
| Average pass length | `total_long_balls` / `total_pass` | Relatively short; wins possession and transfers ball |

---

### Attacking Midfielder (#8)
**What they do:** Connect midfield and attack, progress the ball, exploit half-spaces, support press, and arrive late in the box.

| Metric | Source Column | Notes |
|--------|-----------------|-------|
| Progressive passes per 90 | `total_progression` | Ball progression ability |
| Accurate progressive passes % | `total_progression` / `total_pass` | Ball progression ability |
| Progressive runs per 90 | `progressive_ball_carries_count` | Capable ball carrier |
| xA per 90 | `expected_assists` | Chance-creation ability |
| Shots per 90 | `total_shots` | Not core but enough volume; not shot shy |

---

### Deep-Lying Playmaker
**What they do:** Orchestrate possession from deep areas, switch play, bypass press, and break lines with passing vision.

| Metric | Source Column | Notes |
|--------|-----------------|-------|
| Received passes per 90 | `touches` | High volume suggests trusted outlet |
| Progressive passes per 90 | `total_progression` | Ball progression ability |
| Accurate progressive passes % | `total_progression` / `total_pass` | Ball progression ability |
| Accurate long passes % | `accurate_long_balls` / `total_long_balls` | Passing range |
| Interceptions per 90 | `interception_won` | Reading play early |

---

### Box-to-Box Midfielder
**What they do:** Cover large distances, contribute in both boxes, support transitions, and provide presence during both attacking and defensive phases.

| Metric | Source Column | Notes |
|--------|-----------------|-------|
| Defensive duels per 90 | `duel_won` + `duel_lost` | Defensive contribution |
| Defensive duels won % | `duel_won` / (`duel_won` + `duel_lost`) | Success rate points to ability |
| Progressive runs per 90 | `progressive_ball_carries_count` | Capable ball carrier |
| Touches in box per 90 | `touches` | Above average for position; penalty area presence |
| Shots per 90 | `total_shots` | Above average for position |

---

## Wide Players

### Full-Back
**What they do:** Provide width in deeper zones, support buildup with overlapping runs, defend wide channels, block crosses and recover possession.

| Metric | Source Column | Notes |
|--------|-----------------|-------|
| Defensive duels per 90 | `duel_won` + `duel_lost` | Active defensively; competitive |
| Defensive duels won % | `duel_won` / (`duel_won` + `duel_lost`) | — |
| Interceptions per 90 | `interception_won` | Reading play early |
| Crosses per 90 | `total_cross` | Above-average volume + success rate |
| Accurate crosses % | `accurate_cross` / `total_cross` | Advances to provide attacking threat |

---

### Wing-Back
**What they do:** Hybrid role: provide width and depth, high involvement in final third and recovery runs. Key outlet in systems with three CBs.

| Metric | Source Column | Notes |
|--------|-----------------|-------|
| Interceptions per 90 | `interception_won` | Reading play early |
| Progressive runs per 90 | `progressive_ball_carries_count` | Capable ball carrier |
| Crosses per 90 | `total_cross` | Above-average volume + success rate |
| Accurate crosses % | `accurate_cross` / `total_cross` | Advances to provide attacking threat |
| Shot assists per 90 | `key_pass` | Creates chances |

---

### Classic Winger
**What they do:** Stretch the opposition horizontally, create width, deliver crosses, isolate in 1v1s, and provide assists or secondary assists.

| Metric | Source Column | Notes |
|--------|-----------------|-------|
| Dribbles per 90 | `won_contest` + `challenge_lost` | Takes on and beats opponents |
| Successful dribbles % | `won_contest` / (`won_contest` + `challenge_lost`) | — |
| Progressive runs per 90 | `progressive_ball_carries_count` | Capable ball carrier |
| Crosses per 90 | `total_cross` | Above-average volume + success rate |
| Accurate crosses % | `accurate_cross` / `total_cross` | Suggests classic winger profile |
| Shot assists per 90 | `key_pass` | Creates chances |

---

### Inverted Winger
**What they do:** Start wide, cut inside onto dominant foot, combine with overlapping FB, and shoot or create from central channels or half-spaces.

| Metric | Source Column | Notes |
|--------|-----------------|-------|
| Shots per 90 | `total_shots` | Generates a good volume of shots |
| xG per 90 | `expected_goals` | Above average filters out long-range volume padders |
| Progressive runs per 90 | `progressive_ball_carries_count` | Attacks space with ball at feet |
| Shot assists per 90 | `key_pass` | Creates chances |
| xA per 90 | `expected_assists` | Average and above to capture quality |

---

## Forwards

### Playmaker (#10)
**What they do:** Drop into pockets between midfield and defence, dictate attacking tempo, assist final actions, and support both flanks and strikers.

| Metric | Source Column | Notes |
|--------|-----------------|-------|
| Progressive passes per 90 | `total_progression` | Ball progression ability |
| Accurate progressive passes % | `total_progression` / `total_pass` | Ball progression ability |
| Deep completions per 90 | `accurate_opposition_half_passes` | Links play deep in opposition territory |
| Shot assists per 90 | `key_pass` | Creates chances |
| xA per 90 | `expected_assists` | Of above-average quality |
| Shots per 90 | `total_shots` | Sufficient volume; not shot shy |

---

### Target Man (#9)
**What they do:** Occupy central defenders, hold up play, win aerial duels, lay off for onrushing teammates, and finish inside the box.

| Metric | Source Column | Notes |
|--------|-----------------|-------|
| Received long passes per 90 | `accurate_long_balls` (team) | Outlet for team |
| Won aerial duels % | `aerial_won` / (`aerial_won` + `aerial_lost`) | Competitive in the air |
| Fouls suffered per 90 | `was_fouled` | Shields ball and draws fouls |
| Passes to final third per 90 | `accurate_opposition_half_passes` | Receives and links play |
| xG per 90 | `expected_goals` | Offers goal-scoring threat |

---

### Poacher
**What they do:** Operate on the last line, exploit spaces behind the defence, focus on finishing chances in the box with minimal touches.

| Metric | Source Column | Notes |
|--------|-----------------|-------|
| Touches in box per 90 | `touches` | High volume; penalty box presence |
| Received passes per 90 | `touches` | Below average; runs stretch defence |
| xG per 90 | `expected_goals` | Gets into good positions |
| Non-penalty goals per 90 | `goals` | Scores at an acceptable rate |
| Goal conversion % | `goals` / `total_shots` | Shot volume may not be high but finishes well |

---

### Pressing Forward
**What they do:** Lead the defensive phase from the front, apply constant pressure to defenders and GK, force errors in build-up.

| Metric | Source Column | Notes |
|--------|-----------------|-------|
| Defensive duels per 90 | `duel_won` + `duel_lost` | Well above-average for position |
| Pressing duels per 90 | `won_contest` + `challenge_lost` | Harries opposition defence |
| Interceptions per 90 | `interception_won` | Reading play early |
| Shots per 90 | `total_shots` | Not just a hard-worker; provides attacking output |
| Progressive runs per 90 | `progressive_ball_carries_count` | Capable of transitioning team forward |
