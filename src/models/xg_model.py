"""Custom xG modeli — XGBoost classifier, FBref/StatsBomb event data üzerinde eğitim."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

MODEL_PATH = Path("data/db/xg_model.pkl")

FEATURES = [
    "distance",       # Kaleye mesafe (metre)
    "angle",          # Kale açısı (derece)
    "x", "y",         # Şut koordinatı (0-120, 0-80)
    "body_part",      # Ayak / kafa (encoded)
    "situation",      # Açık oyun / korner / serbest vuruş (encoded)
    "technique",      # Normal / volley / chip (encoded)
    "prev_action",    # Önceki aksiyonun tipi (encoded)
]


def _compute_geometry(shots: pd.DataFrame) -> pd.DataFrame:
    """Koordinatlardan mesafe ve açı hesaplar (StatsBomb 120x80 sahası)."""
    df = shots.copy()
    # Kale merkezi: (120, 40)
    dx = 120 - df["x"].astype(float)
    dy = 40 - df["y"].astype(float)
    df["distance"] = np.sqrt(dx**2 + dy**2)

    # Kale direklerine açı
    goal_y1, goal_y2 = 36, 44
    a = np.arctan2(goal_y2 - df["y"].astype(float), dx)
    b = np.arctan2(goal_y1 - df["y"].astype(float), dx)
    df["angle"] = np.degrees(np.abs(a - b))
    return df


def prepare_features(shots: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Shot event DataFrame'ini model feature'larına çevirir."""
    df = _compute_geometry(shots)

    cat_cols = ["body_part", "situation", "technique", "prev_action"]
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("unknown").astype(str))
        else:
            df[col] = 0

    available = [f for f in FEATURES if f in df.columns]
    X = df[available].fillna(0)

    y = None
    if "outcome" in df.columns:
        y = (df["outcome"].str.lower() == "goal").astype(int)
    elif "goal" in df.columns:
        y = df["goal"].astype(int)

    return X, y


def train(shots: pd.DataFrame, save: bool = True) -> object:
    """XGBoost xG modelini eğitir ve kaydeder."""
    if not XGB_AVAILABLE:
        raise ImportError("xgboost kurulu değil: pip install xgboost")

    X, y = prepare_features(shots)
    if y is None:
        raise ValueError("Eğitim verisi 'outcome' veya 'goal' kolonu içermeli.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model = CalibratedClassifierCV(base, cv=5, method="isotonic")
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    print(f"[xg_model] Brier Score: {brier:.4f} | AUC-ROC: {auc:.4f}")

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"[xg_model] Model kaydedildi: {MODEL_PATH}")

    return model


def predict(shots: pd.DataFrame, model=None) -> pd.Series:
    """Şut DataFrame'i için xG tahminleri döndürür."""
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model bulunamadı. Önce train() çalıştırın.")
        model = joblib.load(MODEL_PATH)

    X, _ = prepare_features(shots)
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=shots.index, name="xg_pred")


def calibration_check(shots: pd.DataFrame, model=None) -> pd.DataFrame:
    """Gerçek gol oranı vs. tahmin edilen xG kalibrasyonunu döndürür."""
    xg = predict(shots, model)
    shots = shots.copy()
    shots["xg_pred"] = xg.values

    y_true = None
    if "outcome" in shots.columns:
        y_true = (shots["outcome"].str.lower() == "goal").astype(int)
    elif "goal" in shots.columns:
        y_true = shots["goal"].astype(int)

    if y_true is None:
        return pd.DataFrame()

    shots["goal_actual"] = y_true
    shots["xg_bin"] = pd.cut(shots["xg_pred"], bins=10)
    cal = shots.groupby("xg_bin", observed=True).agg(
        xg_mean=("xg_pred", "mean"),
        goal_rate=("goal_actual", "mean"),
        count=("goal_actual", "count"),
    ).reset_index()
    return cal
