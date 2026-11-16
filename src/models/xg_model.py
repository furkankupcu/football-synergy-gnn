"""Custom xG model — XGBoost classifier trained on event data."""

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
    "distance",       # distance to goal (metres)
    "angle",          # goal angle (degrees)
    "x", "y",         # shot coordinates (0-120, 0-80)
    "body_part",      # foot / head (encoded)
    "situation",      # open play / corner / free kick (encoded)
    "technique",      # normal / volley / chip (encoded)
    "prev_action",    # type of the previous action (encoded)
]


def _compute_geometry(shots: pd.DataFrame) -> pd.DataFrame:
    """Compute distance and angle from coordinates (StatsBomb 120x80 pitch)."""
    df = shots.copy()
    # Kale merkezi: (120, 40)
    dx = 120 - df["x"].astype(float)
    dy = 40 - df["y"].astype(float)
    df["distance"] = np.sqrt(dx**2 + dy**2)

    # Angle to the goal posts
    goal_y1, goal_y2 = 36, 44
    a = np.arctan2(goal_y2 - df["y"].astype(float), dx)
    b = np.arctan2(goal_y1 - df["y"].astype(float), dx)
    df["angle"] = np.degrees(np.abs(a - b))
    return df


def prepare_features(shots: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Convert a shot event DataFrame into model features."""
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
    """Train and save the XGBoost xG model."""
    if not XGB_AVAILABLE:
        raise ImportError("xgboost is not installed: pip install xgboost")

    X, y = prepare_features(shots)
    if y is None:
        raise ValueError("Training data must contain an 'outcome' or 'goal' column.")

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
    """Return xG predictions for a shot DataFrame."""
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model not found. Run train() first.")
        model = joblib.load(MODEL_PATH)

    X, _ = prepare_features(shots)
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=shots.index, name="xg_pred")


def calibration_check(shots: pd.DataFrame, model=None) -> pd.DataFrame:
    """Return actual goal rate vs predicted xG calibration."""
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
