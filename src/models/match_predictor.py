"""Maç sonucu tahmini — LightGBM multi-class classifier."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

MODEL_PATH = Path("data/db/match_predictor.pkl")

OUTCOME_MAP = {1: "home_win", 0: "draw", -1: "away_win"}

FEATURES = [
    "home_xg_avg5",      # Son 5 maç ev sahibi xG ortalaması
    "away_xg_avg5",      # Son 5 maç deplasman xG ortalaması
    "home_xga_avg5",     # Son 5 maç ev sahibi xGA ortalaması
    "away_xga_avg5",
    "home_ppda_avg5",    # Son 5 maç ev sahibi PPDA ortalaması
    "away_ppda_avg5",
    "home_form5",        # Son 5 maç puan toplamı (W=3, D=1, L=0)
    "away_form5",
    "home_advantage",    # 1 = ev sahibi
]


def _form_points(results: list[str]) -> int:
    """['W','D','L',...] listesinden puan hesaplar."""
    return sum({"W": 3, "D": 1, "L": 0}.get(r, 0) for r in results)


def build_match_features(
    fixtures: pd.DataFrame,
    team_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fikstür ve takım stats'tan maç bazlı feature DataFrame'i oluşturur.
    fixtures: ['date','home_team','away_team','home_goals','away_goals','home_xg','away_xg']
    team_stats: ['team','xg','xga','ppda'] içermeli.
    """
    fixtures = fixtures.sort_values("date").copy()
    rows = []

    for _, match in fixtures.iterrows():
        home, away = match["home_team"], match["away_team"]
        date = match["date"]

        past_home = fixtures[(fixtures["home_team"] == home) & (fixtures["date"] < date)].tail(5)
        past_away = fixtures[(fixtures["away_team"] == away) & (fixtures["date"] < date)].tail(5)

        home_xg_avg = past_home["home_xg"].mean() if "home_xg" in past_home else np.nan
        away_xg_avg = past_away["away_xg"].mean() if "away_xg" in past_away else np.nan
        home_xga_avg = past_home["away_xg"].mean() if "away_xg" in past_home else np.nan
        away_xga_avg = past_away["home_xg"].mean() if "home_xg" in past_away else np.nan

        if "home_goals" in past_home.columns and "away_goals" in past_home.columns:
            home_results = ["W" if r["home_goals"] > r["away_goals"] else "D" if r["home_goals"] == r["away_goals"] else "L"
                           for _, r in past_home.iterrows()]
            away_results = ["W" if r["away_goals"] > r["home_goals"] else "D" if r["home_goals"] == r["away_goals"] else "L"
                           for _, r in past_away.iterrows()]
        else:
            home_results, away_results = [], []

        ts_home = team_stats[team_stats["team"] == home] if "team" in team_stats.columns else pd.DataFrame()
        ts_away = team_stats[team_stats["team"] == away] if "team" in team_stats.columns else pd.DataFrame()

        row = {
            "home_xg_avg5": home_xg_avg,
            "away_xg_avg5": away_xg_avg,
            "home_xga_avg5": home_xga_avg,
            "away_xga_avg5": away_xga_avg,
            "home_ppda_avg5": ts_home["ppda"].iloc[0] if not ts_home.empty and "ppda" in ts_home else np.nan,
            "away_ppda_avg5": ts_away["ppda"].iloc[0] if not ts_away.empty and "ppda" in ts_away else np.nan,
            "home_form5": _form_points(home_results),
            "away_form5": _form_points(away_results),
            "home_advantage": 1,
        }

        if "home_goals" in match and "away_goals" in match:
            hg, ag = match["home_goals"], match["away_goals"]
            if pd.notna(hg) and pd.notna(ag):
                row["outcome"] = 1 if hg > ag else 0 if hg == ag else -1

        rows.append(row)

    return pd.DataFrame(rows)


def train(match_features: pd.DataFrame, save: bool = True) -> object:
    """LightGBM maç tahmin modelini eğitir."""
    if not LGB_AVAILABLE:
        raise ImportError("lightgbm kurulu değil: pip install lightgbm")
    if "outcome" not in match_features.columns:
        raise ValueError("DataFrame 'outcome' kolonu içermeli.")

    df = match_features.dropna(subset=FEATURES + ["outcome"])
    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df["outcome"].map({1: 0, 0: 1, -1: 2})  # ev-galibiyet=0, beraberlik=1, deplasman=2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, proba)
    print(f"[match_predictor] Accuracy: {acc:.4f} | Log-Loss: {ll:.4f}")

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"[match_predictor] Model kaydedildi: {MODEL_PATH}")

    return model


def predict(match_features: pd.DataFrame, model=None) -> pd.DataFrame:
    """Fixture DataFrame'i için sonuç olasılıklarını döndürür."""
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model bulunamadı. Önce train() çalıştırın.")
        model = joblib.load(MODEL_PATH)

    X = match_features[[f for f in FEATURES if f in match_features.columns]].fillna(0)
    proba = model.predict_proba(X)
    result = match_features.copy()
    result["prob_home_win"] = proba[:, 0].round(3)
    result["prob_draw"] = proba[:, 1].round(3)
    result["prob_away_win"] = proba[:, 2].round(3)
    result["predicted"] = model.predict(X)
    result["predicted_label"] = result["predicted"].map({0: "Ev Galibiyeti", 1: "Beraberlik", 2: "Deplasman"})
    return result
