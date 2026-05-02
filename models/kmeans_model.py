"""
models/kmeans_model.py
─────────────────────────────────────────────────────────────────────────────
Customer segmentation using K-Means clustering.

Key upgrades vs original:
  • n_clusters auto-capped to actual unique customer count — no crash on
    small datasets
  • All functions work on copies — original DataFrame never mutated
  • scale_features() returns (scaled_array, scaler) so scaler is reusable
  • Richer feature set: Quantity_Sold, Profit, Avg_Order_Value, Visit_Count
  • find_optimal_clusters() uses elbow method to suggest best k
  • label_clusters() handles any n_clusters (not hardcoded to 3 labels)
  • cluster_summary() includes median and std alongside mean
  • segment_customers() is idempotent — safe to call multiple times
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_N_CLUSTERS = 3
MAX_ELBOW_K        = 8    # upper limit when searching for optimal k
RANDOM_STATE       = 42


# ── 1. Build customer-level features ─────────────────────────────────────────
def prepare_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw transaction rows into one row per customer.

    Features created:
      Quantity_Sold   — total units bought
      Profit          — total profit generated
      Avg_Order_Value — average profit per transaction (spend level)
      Visit_Count     — number of distinct transactions (loyalty proxy)
    """
    required = {"Customer_ID", "Quantity_Sold", "Profit"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"kmeans_model: DataFrame missing columns: {missing}")

    customer_df = (
        df.groupby("Customer_ID")
        .agg(
            Quantity_Sold   = ("Quantity_Sold", "sum"),
            Profit          = ("Profit",        "sum"),
            Avg_Order_Value = ("Profit",        "mean"),
            Visit_Count     = ("Profit",        "count"),
        )
        .reset_index()
    )

    return customer_df


# ── 2. Scale features ─────────────────────────────────────────────────────────
def scale_features(customer_df: pd.DataFrame):
    """
    Normalise clustering features with StandardScaler.
    Returns (scaled_array, fitted_scaler) — scaler returned so it can be
    reused for inverse-transforming or scoring new customers.
    """
    feature_cols = ["Quantity_Sold", "Profit", "Avg_Order_Value", "Visit_Count"]
    available    = [c for c in feature_cols if c in customer_df.columns]

    scaler         = StandardScaler()
    scaled_array   = scaler.fit_transform(customer_df[available])

    return scaled_array, scaler


# ── 3. Optimal k via elbow method ─────────────────────────────────────────────
def find_optimal_clusters(customer_df: pd.DataFrame, max_k: int = MAX_ELBOW_K) -> int:
    """
    Run K-Means for k=2…max_k, pick k at the 'elbow' of the inertia curve.
    Falls back to DEFAULT_N_CLUSTERS if not enough customers.
    """
    n_customers = len(customer_df)
    max_k       = min(max_k, n_customers - 1)  # can't have more clusters than customers

    if max_k < 2:
        return 1

    scaled, _ = scale_features(customer_df)
    inertias  = []

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        km.fit(scaled)
        inertias.append(km.inertia_)

    # Elbow = point of maximum curvature (second derivative)
    if len(inertias) < 2:
        return 2

    deltas      = np.diff(inertias)
    second_diff = np.diff(deltas)
    elbow_idx   = int(np.argmax(np.abs(second_diff))) + 2   # +2 offset: started at k=2
    return elbow_idx


# ── 4. Apply K-Means ──────────────────────────────────────────────────────────
def apply_kmeans(customer_df: pd.DataFrame, n_clusters: int = DEFAULT_N_CLUSTERS):
    """
    Run K-Means.  n_clusters is auto-capped to (unique customers - 1)
    so the model never crashes on small datasets.
    Returns (clustered_df, fitted_kmeans, fitted_scaler).
    """
    customer_df = customer_df.copy()

    # Safety cap
    max_clusters = max(1, len(customer_df) - 1)
    n_clusters   = min(n_clusters, max_clusters)

    if n_clusters < 2:
        customer_df["Cluster"] = 0
        return customer_df, None, None

    scaled, scaler = scale_features(customer_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto")
    customer_df["Cluster"] = kmeans.fit_predict(scaled)

    return customer_df, kmeans, scaler


# ── 5. Full pipeline ──────────────────────────────────────────────────────────
def segment_customers(df: pd.DataFrame, n_clusters: int = DEFAULT_N_CLUSTERS) -> pd.DataFrame:
    """
    Prepare → scale → cluster.  Returns customer DataFrame with Cluster column.
    Safe to call multiple times — works on a copy of df.
    """
    customer_df          = prepare_customer_data(df)
    clustered_df, _, _   = apply_kmeans(customer_df, n_clusters)
    return clustered_df


# ── 6. Cluster summary ────────────────────────────────────────────────────────
def cluster_summary(df: pd.DataFrame, n_clusters: int = DEFAULT_N_CLUSTERS) -> pd.DataFrame:
    """
    Per-cluster statistics: mean, median, std, and customer count.
    """
    customer_df = segment_customers(df, n_clusters)

    summary = (
        customer_df.groupby("Cluster")
        .agg(
            Num_Customers   = ("Customer_ID",    "count"),
            Avg_Quantity    = ("Quantity_Sold",  "mean"),
            Median_Quantity = ("Quantity_Sold",  "median"),
            Avg_Profit      = ("Profit",         "mean"),
            Median_Profit   = ("Profit",         "median"),
            Std_Profit      = ("Profit",         "std"),
            Avg_Visit_Count = ("Visit_Count",    "mean"),
        )
        .round(2)
        .reset_index()
    )

    return summary


# ── 7. Label clusters ─────────────────────────────────────────────────────────
def label_clusters(df: pd.DataFrame, n_clusters: int = DEFAULT_N_CLUSTERS) -> pd.DataFrame:
    """
    Add a human-readable Segment column based on average profit rank.
    Works for any n_clusters — not hardcoded to 3.

    With 3 clusters: Low Value / Medium Value / High Value
    With 2 clusters: Low Value / High Value
    With 4 clusters: Tier 1 (lowest) … Tier 4 (highest)
    """
    customer_df = segment_customers(df, n_clusters)
    profit_avg  = customer_df.groupby("Cluster")["Profit"].mean()
    sorted_clusters = profit_avg.sort_values().index.tolist()

    if n_clusters == 2:
        tier_names = ["Low Value", "High Value"]
    elif n_clusters == 3:
        tier_names = ["Low Value", "Medium Value", "High Value"]
    else:
        tier_names = [f"Tier {i+1}" for i in range(n_clusters)]

    label_map = {cluster: name for cluster, name in zip(sorted_clusters, tier_names)}
    customer_df = customer_df.copy()
    customer_df["Segment"] = customer_df["Cluster"].map(label_map)

    return customer_df


# ── 8. Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils.preprocessing import preprocess_data

    df = preprocess_data()

    print("── Optimal k ────────────────────────────")
    customer_df = prepare_customer_data(df)
    optimal_k   = find_optimal_clusters(customer_df)
    print(f"  Suggested clusters: {optimal_k}\n")

    print("── Segmented Customers (top 10) ─────────")
    segmented = label_clusters(df)
    print(segmented.head(10).to_string(index=False))

    print("\n── Cluster Summary ──────────────────────")
    print(cluster_summary(df).to_string(index=False))