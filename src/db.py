"""PostgreSQL bağlantı yardımcıları — SQLAlchemy + psycopg2."""

import os
from contextlib import contextmanager
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

load_dotenv()


def get_dsn() -> dict:
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME", "footdata"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
    }


def get_engine() -> Engine:
    """SQLAlchemy engine — pandas read_sql/to_sql için."""
    dsn = get_dsn()
    url = (
        f"postgresql+psycopg2://{dsn['user']}:{dsn['password']}"
        f"@{dsn['host']}:{dsn['port']}/{dsn['dbname']}"
    )
    return create_engine(url, pool_pre_ping=True)


@contextmanager
def connect():
    """psycopg2 bağlantısı — context manager olarak."""
    conn = psycopg2.connect(**get_dsn())
    try:
        yield conn
    finally:
        conn.close()


def read_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    """SQL sorgusu çalıştır, DataFrame döndür. Param formatı: :name (SQLAlchemy)."""
    from sqlalchemy import text
    eng = get_engine()
    with eng.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


def list_tables() -> list[str]:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' ORDER BY table_name
        """)
        return [r[0] for r in cur.fetchall()]


def upsert_dataframe(
    df: pd.DataFrame,
    table: str,
    key_cols: list[str],
    delete_first: bool = True,
) -> int:
    """
    DataFrame'i tabloya yazar.
    delete_first=True ise önce key_cols'a göre eski satırları siler,
    sonra yeni satırları INSERT eder (basit replace semantiği).
    """
    if df.empty:
        return 0

    eng = get_engine()

    # Tablodaki gerçek kolonları al, DataFrame'i o kolonlara göre filtrele
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = '{table}' AND table_schema = 'public'
            ORDER BY ordinal_position
        """)
        table_cols = [r[0] for r in cur.fetchall()]

    cols_to_write = [c for c in df.columns if c in table_cols]
    df_filtered = df[cols_to_write].copy()

    # Eski satırları sil
    if delete_first and key_cols:
        valid_keys = [k for k in key_cols if k in df_filtered.columns]
        if valid_keys:
            with connect() as conn:
                cur = conn.cursor()
                for _, row in df_filtered[valid_keys].drop_duplicates().iterrows():
                    where = " AND ".join([f"{k} = %s" for k in valid_keys])
                    cur.execute(
                        f"DELETE FROM {table} WHERE {where}",
                        tuple(row[k] for k in valid_keys),
                    )
                conn.commit()

    df_filtered.to_sql(table, eng, if_exists="append", index=False, method="multi", chunksize=500)
    return len(df_filtered)


def ensure_players(df: pd.DataFrame) -> int:
    """
    DataFrame'deki yeni player_id/player_name çiftlerini `players` tablosuna ekler.
    player_stats_* tablolarındaki FK constraint için gereklidir.
    """
    if "player_id" not in df.columns:
        return 0

    name_col = "player_name" if "player_name" in df.columns else "player"
    if name_col not in df.columns:
        return 0

    new_players = df[["player_id", name_col]].drop_duplicates().dropna(subset=["player_id"])
    new_players = new_players.rename(columns={name_col: "player_name"})

    if new_players.empty:
        return 0

    with connect() as conn:
        cur = conn.cursor()
        inserted = 0
        for _, row in new_players.iterrows():
            cur.execute(
                "INSERT INTO players (player_id, player_name) VALUES (%s, %s) "
                "ON CONFLICT (player_id) DO NOTHING",
                (row["player_id"], row["player_name"]),
            )
            inserted += cur.rowcount
        conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Yüksek seviyeli okuma yardımcıları — analiz modülleri için
# ---------------------------------------------------------------------------

def get_team_stats(league: str | None = None, season: str | None = None) -> pd.DataFrame:
    """
    Tüm team_stats_* tablolarını team_id+season+league üzerinden JOIN edip döndürür.
    """
    where = []
    params = {}
    if league:
        where.append("s.league = :league")
        params["league"] = league
    if season:
        where.append("s.season = :season")
        params["season"] = season
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    query = f"""
        SELECT
            s.team_id, s.team_name, s.league, s.season,
            s.matches_played, s.wins, s.draws, s.losses, s.points,
            s.goals_for, s.goals_against, s.goal_diff,
            s.xg, s.xg_against, s.xg_diff,
            sh.shots, sh.shots_on_target, sh.shots_on_target_pct,
            sh.shots_per90, sh.average_shot_distance, sh.npxg,
            p.passes_completed, p.passes, p.passes_pct,
            p.passes_progressive_distance, p.assists AS pass_assists,
            p.xg_assist, p.assisted_shots, p.passes_into_final_third,
            p.passes_into_penalty_area, p.progressive_passes,
            d.tackles, d.tackles_won, d.blocks, d.interceptions,
            d.tackles_interceptions, d.clearances, d.errors,
            poss.possession, poss.touches, poss.progressive_carries,
            poss.carries, poss.take_ons_attempted, poss.take_ons_won
        FROM team_stats_standard s
        LEFT JOIN team_stats_shooting sh
            ON sh.team_id = s.team_id AND sh.season = s.season AND sh.league = s.league
        LEFT JOIN team_stats_passing p
            ON p.team_id = s.team_id AND p.season = s.season AND p.league = s.league
        LEFT JOIN team_stats_defense d
            ON d.team_id = s.team_id AND d.season = s.season AND d.league = s.league
        LEFT JOIN team_stats_possession poss
            ON poss.team_id = s.team_id AND poss.season = s.season AND poss.league = s.league
        {where_sql}
    """
    return read_sql(query, params)


def get_player_stats(league: str | None = None, season: str | None = None) -> pd.DataFrame:
    """
    Tüm player_stats_* tablolarını + players + tm_bio'yu birleştirir.
    """
    where = []
    params = {}
    if league:
        where.append("s.league = :league")
        params["league"] = league
    if season:
        where.append("s.season = :season")
        params["season"] = season
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    query = f"""
        SELECT
            pl.player_name AS player,
            s.player_id, s.team, s.league, s.season,
            s.position, s.age, s.matches_played AS mp, s.starts, s.minutes,
            s.goals, s.assists, s.goals_assists, s.goals_pens,
            s.xg, s.npxg, s.xg_assist AS xa, s.npxg_xg_assist,
            s.progressive_carries, s.progressive_passes,
            sh.shots, sh.shots_on_target, sh.shots_per90,
            pa.passes_completed, pa.passes, pa.passes_pct,
            pa.assisted_shots, pa.passes_into_final_third,
            pa.pass_xa, pa.crosses_into_penalty_area,
            d.tackles, d.tackles_won, d.interceptions,
            d.tackles_interceptions, d.blocks, d.clearances,
            poss.touches, poss.carries, poss.take_ons_attempted,
            poss.take_ons_won, poss.miscontrols, poss.dispossessed,
            m.cards_yellow, m.cards_red, m.fouls, m.fouled,
            m.aerials_won, m.aerials_lost, m.aerials_won_pct,
            tm.market_value AS market_value_raw,
            tm.main_position AS tm_position,
            tm.height, tm.foot, tm.citizenship
        FROM player_stats_standard s
        LEFT JOIN players pl ON pl.player_id = s.player_id
        LEFT JOIN player_stats_shooting sh
            ON sh.player_id = s.player_id AND sh.season = s.season AND sh.league = s.league
        LEFT JOIN player_stats_passing pa
            ON pa.player_id = s.player_id AND pa.season = s.season AND pa.league = s.league
        LEFT JOIN player_stats_defense d
            ON d.player_id = s.player_id AND d.season = s.season AND d.league = s.league
        LEFT JOIN player_stats_possession poss
            ON poss.player_id = s.player_id AND poss.season = s.season AND poss.league = s.league
        LEFT JOIN player_stats_misc m
            ON m.player_id = s.player_id AND m.season = s.season AND m.league = s.league
        LEFT JOIN player_tm_bio tm ON tm.player_name = pl.player_name
        {where_sql}
    """
    df = read_sql(query, params)

    # Transfermarkt market_value'yu float'a çevir
    if "market_value_raw" in df.columns:
        df["market_value"] = df["market_value_raw"].apply(_parse_market_value)

    return df


def _parse_market_value(val) -> float | None:
    """'€12.00m' / '€500k' → float."""
    if not isinstance(val, str):
        return None
    s = val.replace("€", "").replace(",", "").strip()
    if s == "-" or not s:
        return None
    try:
        if s.endswith("m"):
            return float(s[:-1]) * 1_000_000
        if s.endswith("k"):
            return float(s[:-1]) * 1_000
        return float(s)
    except ValueError:
        return None


def get_player_stats_from_source(
    league: str | None = None,
    season: str | None = None,
) -> pd.DataFrame:
    """
    player_stats → sezon başına kümülatif oyuncu istatistikleri.
    player_id: 'ss_<source_player_id>' (match_lineups ile uyumlu).
    season çıktısı: '2025-2026' formatı (match_lineups ile uyumlu).
    season girişi hem '25/26' hem '2025-2026' formatını kabul eder.
    """
    where = []
    params: dict = {}
    if league:
        where.append("m.league = :league")
        params["league"] = league
    if season:
        parts = str(season).split("-")
        if len(parts) == 2 and len(parts[0]) == 4:
            # '2025-2026' → '25/26'
            ss_season = f"{parts[0][2:]}/{parts[1][2:]}"
        else:
            ss_season = season
        where.append("(m.season = :season_full OR m.season = :season_ss)")
        params["season_full"] = season
        params["season_ss"] = ss_season

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    query = f"""
        SELECT
            'ss_' || p.source_player_id::text        AS player_id,
            p.player_name,
            p.team_name                                  AS team,
            m.league,
            CASE
                WHEN m.season LIKE '__/__' THEN
                    '20' || SPLIT_PART(m.season, '/', 1) ||
                    '-20' || SPLIT_PART(m.season, '/', 2)
                ELSE m.season
            END                                          AS season,
            MODE() WITHIN GROUP (ORDER BY p.position)   AS position,
            SUM(COALESCE(p.minutes_played, 0))           AS minutes,
            SUM(COALESCE(p.goals, 0))                    AS goals,
            SUM(COALESCE(p.goal_assist, 0))              AS assists,
            SUM(COALESCE(p.expected_goals, 0))           AS xg,
            SUM(COALESCE(p.expected_goals, 0))           AS npxg,
            SUM(COALESCE(p.expected_assists, 0))         AS xa,
            SUM(COALESCE(p.total_shots, 0))              AS shots,
            SUM(COALESCE(p.on_target_scoring_attempt, 0)) AS shots_on_target,
            SUM(COALESCE(p.total_tackle, 0))             AS tackles,
            SUM(COALESCE(p.won_tackle, 0))               AS tackles_won,
            SUM(COALESCE(p.interception_won, 0))         AS interceptions,
            SUM(COALESCE(p.outfielder_block, 0))         AS blocks,
            SUM(COALESCE(p.total_clearance, 0))          AS clearances,
            SUM(COALESCE(p.accurate_pass, 0))            AS passes_completed,
            SUM(COALESCE(p.total_pass, 0))               AS passes,
            CASE
                WHEN SUM(COALESCE(p.total_pass, 0)) > 0
                THEN ROUND(
                    SUM(COALESCE(p.accurate_pass, 0))::numeric /
                    SUM(COALESCE(p.total_pass, 0)) * 100, 1)
                ELSE 0
            END                                          AS passes_pct,
            SUM(COALESCE(p.key_pass, 0))                 AS assisted_shots,
            SUM(COALESCE(p.accurate_opposition_half_passes, 0)) AS passes_into_final_third,
            SUM(COALESCE(p.touches, 0))                  AS touches,
            SUM(COALESCE(p.ball_carries_count, 0))       AS carries,
            SUM(COALESCE(p.progressive_ball_carries_count, 0)) AS progressive_carries,
            SUM(COALESCE(p.total_contest, 0))            AS take_ons_attempted,
            SUM(COALESCE(p.won_contest, 0))              AS take_ons_won,
            SUM(COALESCE(p.aerial_won, 0))               AS aerials_won,
            SUM(COALESCE(p.aerial_lost, 0))              AS aerials_lost,
            SUM(COALESCE(p.fouls, 0))                    AS fouls,
            SUM(COALESCE(p.was_fouled, 0))               AS fouled,
            0                                            AS cards_yellow,
            0                                            AS cards_red,
            0                                            AS progressive_passes,
            0                                            AS passes_into_penalty_area,
            NULL::int                                    AS age,
            NULL::int                                    AS height,
            NULL::text                                   AS foot,
            NULL::text                                   AS tm_position
        FROM player_stats p
        JOIN matches m ON m.match_id = p.match_id
        {where_sql}
        GROUP BY p.source_player_id, p.player_name, p.team_name, m.league, m.season
    """
    return read_sql(query, params)


def get_available_leagues(season: str | None = None) -> list[str]:
    """Veritabanındaki mevcut ligleri döndürür."""
    where = "WHERE season = :season" if season else ""
    query = f"SELECT DISTINCT league FROM team_stats_standard {where} ORDER BY league"
    df = read_sql(query, {"season": season} if season else None)
    return df["league"].tolist()


def get_available_seasons() -> list[str]:
    """Veritabanındaki mevcut sezonları döndürür."""
    query = "SELECT DISTINCT season FROM team_stats_standard ORDER BY season DESC"
    df = read_sql(query)
    return df["season"].tolist()
