"""
models/apriori_model.py
─────────────────────────────────────────────────────────────────────────────
Market basket analysis using the Apriori algorithm.

Key upgrades vs original:
  • Accepts the preprocessed DataFrame directly — no separate CSV load needed
  • Falls back to transaction_dataset.csv only if no df is passed
  • Validates required columns before processing
  • Ensures basket values are strictly boolean (0/1) for mlxtend compatibility
  • Configurable min_support, min_confidence, min_lift via constants
  • Rules cached in memory so repeated calls in the same session are instant
  • recommend_for_product() fixed — correctly matches frozenset antecedents
  • top_rules() helper returns cleaned, sorted DataFrame ready for display
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_SUPPORT    = 0.01   # item must appear in ≥1% of transactions
MIN_CONFIDENCE = 0.1    # rule must be correct ≥10% of the time
MIN_LIFT       = 1.0    # only rules stronger than random chance

REQUIRED_COLS  = {"Transaction_ID", "Product_Name"}
FALLBACK_PATH  = "data/transaction_dataset.csv"

# ── Module-level cache ────────────────────────────────────────────────────────
_rules_cache: pd.DataFrame | None = None


# ── 1. Load / receive data ────────────────────────────────────────────────────
def load_transaction_data(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Returns a transaction DataFrame.

    Priority:
      1. Use `df` if passed and it contains Transaction_ID & Product_Name.
      2. Fall back to reading FALLBACK_PATH from disk.
    """
    if df is not None and not df.empty:
        missing = REQUIRED_COLS - set(df.columns)
        if not missing:
            return df[list(REQUIRED_COLS)].dropna()
        # df exists but lacks the right columns → fall through to CSV

    try:
        csv_df = pd.read_csv(FALLBACK_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Transaction data not found at '{FALLBACK_PATH}'. "
            "Pass a DataFrame with 'Transaction_ID' and 'Product_Name' columns."
        )

    missing = REQUIRED_COLS - set(csv_df.columns)
    if missing:
        raise ValueError(f"Transaction CSV is missing columns: {missing}")

    return csv_df[list(REQUIRED_COLS)].dropna()


# ── 2. Build basket matrix ────────────────────────────────────────────────────
def create_basket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts transaction rows → boolean (True/False) transaction × product matrix.
    mlxtend's apriori requires boolean dtype — not int.
    """
    basket = df.pivot_table(
        index="Transaction_ID",
        columns="Product_Name",
        aggfunc=lambda x: True,
        fill_value=False,
    )

    # Flatten MultiIndex columns if pivot_table created one
    if isinstance(basket.columns, pd.MultiIndex):
        basket.columns = basket.columns.get_level_values(-1)

    # Ensure strictly boolean
    return basket.astype(bool)


# ── 3. Frequent itemsets ──────────────────────────────────────────────────────
def generate_frequent_itemsets(
    basket: pd.DataFrame,
    min_support: float = MIN_SUPPORT,
) -> pd.DataFrame:
    if basket.empty:
        return pd.DataFrame()

    freq = apriori(basket, min_support=min_support, use_colnames=True, low_memory=False)
    return freq


# ── 4. Association rules ──────────────────────────────────────────────────────
def generate_rules(
    freq_items: pd.DataFrame,
    min_confidence: float = MIN_CONFIDENCE,
    min_lift: float = MIN_LIFT,
) -> pd.DataFrame:
    if freq_items.empty:
        return pd.DataFrame()

    rules = association_rules(
        freq_items, metric="confidence", min_threshold=min_confidence
    )

    # Apply lift filter after generation
    rules = rules[rules["lift"] >= min_lift].copy()

    # Keep only useful columns, sort by lift
    keep = ["antecedents", "consequents", "support", "confidence", "lift"]
    rules = rules[keep].sort_values("lift", ascending=False).reset_index(drop=True)

    return rules


# ── 5. Main pipeline ──────────────────────────────────────────────────────────
def get_rules(
    df: pd.DataFrame | None = None,
    min_support: float = MIN_SUPPORT,
    min_confidence: float = MIN_CONFIDENCE,
    min_lift: float = MIN_LIFT,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Full Apriori pipeline.  Returns association rules DataFrame.

    Parameters
    ----------
    df             : Optional preprocessed DataFrame (uses Transaction_ID + Product_Name).
    min_support    : Minimum support threshold.
    min_confidence : Minimum confidence threshold.
    min_lift       : Minimum lift threshold.
    use_cache      : Return cached rules if available (avoids recomputing).
    """
    global _rules_cache

    if use_cache and _rules_cache is not None:
        return _rules_cache

    try:
        data = load_transaction_data(df)
    except (FileNotFoundError, ValueError) as e:
        print(f"⚠️  Apriori data error: {e}")
        return pd.DataFrame()

    if data.empty:
        print("⚠️  Transaction data is empty — no rules generated.")
        return pd.DataFrame()

    basket    = create_basket(data)
    freq      = generate_frequent_itemsets(basket, min_support)
    rules     = generate_rules(freq, min_confidence, min_lift)

    if rules.empty:
        print(
            f"⚠️  No rules generated with support={min_support}, "
            f"confidence={min_confidence}, lift≥{min_lift}. "
            "Try lowering MIN_SUPPORT at the top of apriori_model.py."
        )

    _rules_cache = rules
    return rules


def clear_cache():
    """Force re-computation on next get_rules() call."""
    global _rules_cache
    _rules_cache = None


# ── 6. Cleaned rules for display ─────────────────────────────────────────────
def top_rules(df: pd.DataFrame | None = None, n: int = 10) -> pd.DataFrame:
    """
    Returns top-N rules as a clean display DataFrame with string columns
    (frozensets converted to readable text).
    """
    rules = get_rules(df)
    if rules.empty:
        return pd.DataFrame(columns=["If customer buys", "They also buy", "Support", "Confidence", "Lift"])

    display = rules.head(n).copy()
    display["antecedents"] = display["antecedents"].apply(lambda x: " + ".join(sorted(x)))
    display["consequents"] = display["consequents"].apply(lambda x: " + ".join(sorted(x)))
    display = display.rename(columns={
        "antecedents": "If customer buys",
        "consequents": "They also buy",
        "support":     "Support",
        "confidence":  "Confidence",
        "lift":        "Lift",
    })
    display[["Support", "Confidence", "Lift"]] = display[
        ["Support", "Confidence", "Lift"]
    ].round(3)

    return display.reset_index(drop=True)


# ── 7. Product-specific recommendations ──────────────────────────────────────
def recommend_for_product(product_name: str, df: pd.DataFrame | None = None) -> list[str]:
    """
    Returns a list of products frequently bought with `product_name`.
    Fixed: correctly checks frozenset membership.
    """
    rules = get_rules(df)
    if rules.empty:
        return []

    recommendations = []
    for _, row in rules.iterrows():
        if product_name in row["antecedents"]:                  # frozenset `in` check ✅
            recommendations.extend(list(row["consequents"]))

    # Deduplicate and remove the input product itself
    seen = set()
    result = []
    for p in recommendations:
        if p != product_name and p not in seen:
            seen.add(p)
            result.append(p)

    return result


# ── 8. Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rules = get_rules()

    if rules.empty:
        print("No rules generated ❌")
    else:
        print(f"✅ {len(rules)} association rules generated\n")
        print(top_rules())

        # Test product recommendation
        sample_product = list(rules["antecedents"].iloc[0])[0]
        recs = recommend_for_product(sample_product)
        print(f"\n🛒 If customer buys '{sample_product}', recommend: {recs}")