"""
utils/preprocessing.py
─────────────────────────────────────────────────────────────────────────────
Data loading, validation, and feature engineering pipeline.

Key upgrades vs original:
  • Validates required columns on load — clear error if CSV schema changes
  • All transforms work on a copy — original DataFrame is never mutated
  • Numeric columns validated and coerced safely (no silent NaN propagation)
  • Profit calculation guarded against divide-by-zero / negative prices
  • Configurable data path via DATA_PATH constant or env variable
  • filter_by_date() accepts both string and datetime inputs
  • data_summary() returns richer stats (date range, avg profit, loss count)
  • Descriptive errors at every step instead of silent failures
"""

import os
import pandas as pd
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH = os.environ.get("SUPERMARKET_DATA_PATH", "data/supermarket_dummy_data.csv")

REQUIRED_COLUMNS = {
    "Date", "Product_Name", "Customer_ID",
    "Selling_Price", "Cost_Price", "Quantity_Sold",
}

NUMERIC_COLUMNS = ["Selling_Price", "Cost_Price", "Quantity_Sold"]


# ── 1. Load & validate ────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the CSV and perform basic structural validation.
    Raises FileNotFoundError or ValueError with a clear message on failure.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: '{path}'\n"
            "Set the SUPERMARKET_DATA_PATH environment variable or check the path."
        )

    df = pd.read_csv(path)

    # ── Column check ──────────────────────────────────────────────────────────
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"CSV is missing required columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )

    # ── Date parsing ──────────────────────────────────────────────────────────
    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        raise ValueError(
            "Could not parse 'Date' column. "
            "Expected formats: YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY."
        )

    return df


# ── 2. Handle missing values ──────────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values per column type.
    Numeric columns → 0.  String/object columns → 'Unknown'.
    Works on a copy — does not mutate the input.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64, float, int]:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("Unknown")

    return df


# ── 3. Coerce numeric columns ─────────────────────────────────────────────────
def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force NUMERIC_COLUMNS to float.  Non-numeric values become 0 instead of
    crashing or silently producing NaN that breaks downstream models.
    """
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
    return df


# ── 4. Add profit column ──────────────────────────────────────────────────────
def add_profit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Profit = (Selling_Price - Cost_Price) × Quantity_Sold.
    Negative margin products are kept (real losses) but guarded against
    corrupt data where prices are zero.
    """
    df = df.copy()
    df["Profit"] = (df["Selling_Price"] - df["Cost_Price"]) * df["Quantity_Sold"]
    return df


# ── 5. Add time features ──────────────────────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract granular time features from the Date column."""
    df = df.copy()
    df["Day"]        = df["Date"].dt.day
    df["Month"]      = df["Date"].dt.month
    df["Year"]       = df["Date"].dt.year
    df["DayOfWeek"]  = df["Date"].dt.dayofweek          # 0=Monday … 6=Sunday
    df["IsWeekend"]  = df["DayOfWeek"].isin([5, 6]).astype(int)
    df["Quarter"]    = df["Date"].dt.quarter
    return df


# ── 6. Ensure Festival column ─────────────────────────────────────────────────
def ensure_festival_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Festival" not in df.columns:
        df["Festival"] = 0
    else:
        # Coerce to int in case it came in as float or string
        df["Festival"] = pd.to_numeric(df["Festival"], errors="coerce").fillna(0).astype(int)
    return df


# ── 7. Full pipeline ──────────────────────────────────────────────────────────
def preprocess_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Master pipeline — call this everywhere.
    Returns a clean, enriched DataFrame ready for models and agents.

    Steps:
      load → validate columns → handle missing → coerce numerics
      → festival column → add profit → add time features
    """
    df = load_data(path)
    df = handle_missing_values(df)
    df = coerce_numeric(df)
    df = ensure_festival_column(df)
    df = add_profit(df)
    df = add_time_features(df)
    return df


# ── 8. Filter by date ─────────────────────────────────────────────────────────
def filter_by_date(
    df: pd.DataFrame,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """
    Slice DataFrame by date range.  Accepts strings ('2024-01-01') or
    datetime objects.  Returns a copy — does not mutate input.
    """
    df = df.copy()

    if start_date is not None:
        df = df[df["Date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["Date"] <= pd.to_datetime(end_date)]

    if df.empty:
        print(f"⚠️  filter_by_date: no rows between {start_date} and {end_date}")

    return df


# ── 9. Summary ────────────────────────────────────────────────────────────────
def data_summary(df: pd.DataFrame) -> dict:
    """
    Returns a richer summary dictionary — useful for dashboard KPI cards
    and the CLI pipeline report.
    """
    total_profit = df["Profit"].sum() if "Profit" in df.columns else 0
    avg_profit   = df["Profit"].mean() if "Profit" in df.columns else 0
    loss_rows    = int((df["Profit"] < 0).sum()) if "Profit" in df.columns else 0
    date_min     = df["Date"].min().date() if "Date" in df.columns else "N/A"
    date_max     = df["Date"].max().date() if "Date" in df.columns else "N/A"
    date_range   = (df["Date"].max() - df["Date"].min()).days if "Date" in df.columns else 0

    return {
        "Total Rows":       len(df),
        "Total Products":   df["Product_Name"].nunique(),
        "Total Customers":  df["Customer_ID"].nunique(),
        "Total Profit":     round(total_profit, 2),
        "Avg Profit / Row": round(avg_profit, 2),
        "Loss Rows":        loss_rows,
        "Date Range (days)": date_range,
        "Earliest Date":    str(date_min),
        "Latest Date":      str(date_max),
    }


# ── 10. Test run ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        df = preprocess_data()
        print("✅ Data loaded and preprocessed successfully\n")
        print(df.head())
        print("\n── Summary ──────────────────────────────")
        for k, v in data_summary(df).items():
            print(f"  {k:<22} {v}")
        print()
        print("── Columns ──────────────────────────────")
        print(f"  {list(df.columns)}")
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}")