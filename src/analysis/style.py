"""Playing-style analysis — PCA dimensionality reduction and K-Means clustering."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Actual columns of the PostgreSQL get_team_stats output
STYLE_FEATURES = [
    "goals_for", "goals_against", "xg", "xg_against",
    "possession", "passes_pct",
    "shots", "shots_on_target",
    "progressive_passes", "progressive_carries",
    "passes_into_final_third", "passes_into_penalty_area",
    "tackles", "interceptions", "blocks",
    "take_ons_attempted", "take_ons_won",
    "average_shot_distance",
]


def prepare_style_matrix(team_stats: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepare the model feature matrix from a team stats DataFrame."""
    available = [f for f in STYLE_FEATURES if f in team_stats.columns]
    if not available:
        raise ValueError("No usable style feature found.")

    X = team_stats[available].apply(pd.to_numeric, errors="coerce")
    # Drop a feature if too much of it is missing
    X = X.dropna(axis=1, thresh=int(len(X) * 0.5))
    X = X.fillna(X.median(numeric_only=True))
    return X, list(X.columns)


def fit_pca(X: pd.DataFrame, variance_threshold: float = 0.85) -> tuple[pd.DataFrame, PCA, StandardScaler]:
    """PCA: choose the number of components that keeps variance_threshold of the variance."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = max(2, int(np.searchsorted(cumvar, variance_threshold)) + 1)
    n_components = min(n_components, X.shape[1])

    pca_final = PCA(n_components=n_components)
    X_pca = pca_final.fit_transform(X_scaled)

    cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=cols, index=X.index)
    return df_pca, pca_final, scaler


def find_optimal_k(X_pca: pd.DataFrame, k_range: range = range(2, 8)) -> int:
    """Find the optimal number of clusters by silhouette score."""
    best_k, best_score = 2, -1
    for k in k_range:
        if k >= len(X_pca):
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def cluster_teams(
    team_stats: pd.DataFrame,
    n_clusters: int | None = None,
    variance_threshold: float = 0.85,
) -> pd.DataFrame:
    """Cluster teams by playing style; returns team_stats + cluster + PC1 + PC2."""
    X, _ = prepare_style_matrix(team_stats)
    X_pca, _, _ = fit_pca(X, variance_threshold)

    k = n_clusters or find_optimal_k(X_pca)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)

    result = team_stats.loc[X.index].copy()
    result["cluster"] = labels
    result["PC1"] = X_pca["PC1"].values
    result["PC2"] = X_pca["PC2"].values if "PC2" in X_pca.columns else 0.0
    return result


def style_fingerprint(
    team_stats: pd.DataFrame,
    team_name: str,
    name_col: str = "team_name",
) -> pd.Series:
    """Return one team's style fingerprint as league percentiles."""
    X, features = prepare_style_matrix(team_stats)
    percentiles = (X.rank(pct=True) * 100).round(1)

    mask = team_stats[name_col] == team_name
    if not mask.any():
        return pd.Series(dtype=float)

    idx = team_stats[mask].index[0]
    if idx not in percentiles.index:
        return pd.Series(dtype=float)
    return percentiles.loc[idx, features]


def describe_cluster(clustered: pd.DataFrame, cluster_id: int) -> pd.Series:
    """Return the average profile of the teams in a cluster."""
    cluster_teams_df = clustered[clustered["cluster"] == cluster_id]
    numeric_cols = cluster_teams_df.select_dtypes(include=[np.number]).columns
    return cluster_teams_df[numeric_cols].mean()
