-- Maç bazlı veri şeması
-- Tablolar: matches, player_stats, shots, team_stats

CREATE TABLE IF NOT EXISTS matches (
    match_id            TEXT PRIMARY KEY,           -- 'ss_<source_id>'
    source_id        BIGINT UNIQUE NOT NULL,
    league              TEXT NOT NULL,
    season              TEXT NOT NULL,
    match_date          DATE,
    round               INT,
    home_team_id        BIGINT,
    home_team_name      TEXT,
    away_team_id        BIGINT,
    away_team_name      TEXT,
    home_goals          INT,
    away_goals          INT,
    home_goals_ht       INT,
    away_goals_ht       INT,
    winner_code         INT,                        -- 1=ev sahibi, 2=berabere, 3=deplasman
    has_xg              BOOLEAN DEFAULT FALSE,
    scraped_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ss_matches_league_season ON matches (league, season);

-- -----------------------------------------------------------------------
-- Maç bazlı oyuncu istatistikleri (tüm stat'lar)
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS player_stats (
    match_id                            TEXT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    source_player_id                 BIGINT NOT NULL,
    player_name                         TEXT,
    team_id                             BIGINT,
    team_name                           TEXT,

    -- Maç bağlamı
    position                            TEXT,           -- G/D/M/F
    substitute                          BOOLEAN,
    minutes_played                      INT,
    jersey_number                       INT,
    captain                             BOOLEAN DEFAULT FALSE,
    rating                              DOUBLE PRECISION,

    -- Şut & gol
    goals                               INT,
    total_shots                         INT,
    on_target_scoring_attempt           INT,
    shot_off_target                     INT,
    blocked_scoring_attempt             INT,
    hit_woodwork                        INT,
    expected_goals                      DOUBLE PRECISION,
    expected_goals_on_target            DOUBLE PRECISION,
    big_chance_created                  INT,

    -- Asist & pas
    goal_assist                         INT,
    expected_assists                    DOUBLE PRECISION,
    key_pass                            INT,
    total_pass                          INT,
    accurate_pass                       INT,
    total_long_balls                    INT,
    accurate_long_balls                 INT,
    total_cross                         INT,
    accurate_cross                      INT,
    accurate_own_half_passes            INT,
    total_own_half_passes               INT,
    accurate_opposition_half_passes     INT,
    total_opposition_half_passes        INT,

    -- Defans
    total_tackle                        INT,
    won_tackle                          INT,
    interception_won                    INT,
    total_clearance                     INT,
    outfielder_block                    INT,
    last_man_tackle                     INT,
    duel_won                            INT,
    duel_lost                           INT,
    aerial_won                          INT,
    aerial_lost                         INT,
    challenge_lost                      INT,
    won_contest                         INT,
    total_contest                       INT,
    error_lead_to_a_shot                INT,

    -- Top taşıma & ilerleme
    ball_carries_count                  INT,
    progressive_ball_carries_count      INT,
    total_ball_carries_distance         DOUBLE PRECISION,
    total_progressive_ball_carries_distance DOUBLE PRECISION,
    total_progression                   DOUBLE PRECISION,
    best_ball_carry_progression         DOUBLE PRECISION,

    -- Genel
    touches                             INT,
    ball_recovery                       INT,
    possession_lost_ctrl                INT,
    dispossessed                        INT,
    unsuccessful_touch                  INT,
    total_offside                       INT,
    fouls                               INT,
    was_fouled                          INT,

    -- Kaleci
    saves                               INT,
    saved_shots_from_inside_the_box     INT,
    goals_prevented                     DOUBLE PRECISION,
    keeper_save_value                   DOUBLE PRECISION,
    total_keeper_sweeper                INT,
    accurate_keeper_sweeper             INT,

    -- Normalize değer skorları
    pass_value_normalized               DOUBLE PRECISION,
    dribble_value_normalized            DOUBLE PRECISION,
    defensive_value_normalized          DOUBLE PRECISION,
    goalkeeper_value_normalized         DOUBLE PRECISION,
    shot_value_normalized               DOUBLE PRECISION,

    PRIMARY KEY (match_id, source_player_id)
);

CREATE INDEX IF NOT EXISTS idx_ss_player_stats_player ON player_stats (source_player_id);
CREATE INDEX IF NOT EXISTS idx_ss_player_stats_match  ON player_stats (match_id);
CREATE INDEX IF NOT EXISTS idx_ss_player_stats_team   ON player_stats (team_id);

-- -----------------------------------------------------------------------
-- Şut bazlı xG (koordinatlarla)
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS shots (
    id                      BIGINT,
    match_id                TEXT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    time_minute             INT,
    added_time              INT,
    time_seconds            INT,
    is_home                 BOOLEAN,
    shot_type               TEXT,           -- 'goal', 'miss', 'save', 'block', 'woodwork'
    goal_type               TEXT,
    situation               TEXT,           -- 'openPlay', 'setpiece', 'corner', 'freekick', 'penalty'
    body_part               TEXT,           -- 'foot', 'head', 'otherBodyPart'
    xg                      DOUBLE PRECISION,
    xgot                    DOUBLE PRECISION,
    player_name             TEXT,
    goalkeeper_name         TEXT,
    player_coord_x          DOUBLE PRECISION,
    player_coord_y          DOUBLE PRECISION,
    goal_mouth_x            DOUBLE PRECISION,
    goal_mouth_y            DOUBLE PRECISION,
    block_coord_x           DOUBLE PRECISION,
    block_coord_y           DOUBLE PRECISION,
    PRIMARY KEY (match_id, id)
);

CREATE INDEX IF NOT EXISTS idx_ss_shots_match ON shots (match_id);

-- -----------------------------------------------------------------------
-- Maç bazlı takım istatistikleri (122 stat, pivotlanmış)
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS team_stats (
    match_id            TEXT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    team_side           TEXT NOT NULL,      -- 'home' | 'away'
    period              TEXT NOT NULL,      -- 'ALL' | '1ST' | '2ND'

    -- Genel
    ball_possession         DOUBLE PRECISION,
    expected_goals          DOUBLE PRECISION,
    big_chances             INT,
    big_chances_missed      INT,
    total_shots             INT,
    shots_on_target         INT,
    shots_off_target        INT,
    blocked_shots           INT,
    hit_woodwork            INT,
    shots_inside_box        INT,
    shots_outside_box       INT,

    -- Pas
    total_passes            INT,
    accurate_passes         INT,
    accurate_passes_pct     DOUBLE PRECISION,
    total_long_balls        INT,
    accurate_long_balls     INT,
    total_crosses           INT,
    accurate_crosses        INT,
    key_passes              INT,

    -- Defans
    total_tackles           INT,
    interceptions           INT,
    clearances              INT,
    duel_won                INT,
    duel_lost               INT,
    aerial_won              INT,
    aerial_lost             INT,
    errors_to_shot          INT,

    -- Baskı
    tackles_won             INT,
    def_actions             INT,
    recoveries              INT,
    opponents_half_recoveries INT,

    -- Disiplin
    fouls                   INT,
    offsides                INT,
    yellow_cards            INT,
    red_cards               INT,
    corner_kicks            INT,

    PRIMARY KEY (match_id, team_side, period)
);
