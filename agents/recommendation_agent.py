"""
agents/recommendation_agent.py
─────────────────────────────────────────────────────────────────────────────
Product recommendation agent powered by Apriori association rules.

Key upgrades vs original:
  • Accepts df so it works with the upgraded apriori_model (no separate CSV)
  • Uses top_rules() and recommend_for_product() from upgraded apriori_model
  • formatted_rules() and top_recommendations() work on copies — no mutation
  • recommend_for_product() now returns flat deduplicated list (was nested)
  • recommendation_insights() generates specific data-driven insights
    instead of two hardcoded generic strings
  • recommendation_agent_summary() accepts df and passes it through
  • All functions handle empty rules gracefully
"""

import pandas as pd
from models.apriori_model import get_rules, top_rules, recommend_for_product as _apriori_recommend


# ── 1. Raw rules ──────────────────────────────────────────────────────────────
def get_all_rules(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return raw association rules DataFrame (frozensets intact)."""
    return get_rules(df)


# ── 2. Formatted rules ────────────────────────────────────────────────────────
def formatted_rules(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Return rules with frozensets converted to readable strings.
    Works on a copy — does not mutate cached rules.
    """
    rules = get_rules(df)
    if rules.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    display = rules.copy()
    display["antecedents"] = display["antecedents"].apply(lambda x: " + ".join(sorted(x)))
    display["consequents"] = display["consequents"].apply(lambda x: " + ".join(sorted(x)))

    return display[["antecedents", "consequents", "support", "confidence", "lift"]].reset_index(drop=True)


# ── 3. Top recommendations ────────────────────────────────────────────────────
def top_recommendations(df: pd.DataFrame | None = None, top_n: int = 5) -> pd.DataFrame:
    """
    Return top-N rules sorted by lift with human-readable column names.
    Delegates to the upgraded top_rules() helper in apriori_model.
    """
    return top_rules(df, n=top_n)


# ── 4. Recommend products for a given item ────────────────────────────────────
def recommend_for_product(product_name: str, df: pd.DataFrame | None = None) -> list[str]:
    """
    Return a flat deduplicated list of products to recommend when
    a customer picks up `product_name`.

    Fixed vs original:
      - Returns a flat list, not a nested list-of-lists
      - Delegates to apriori_model's fixed frozenset-aware implementation
    """
    return _apriori_recommend(product_name, df)


# ── 5. Data-driven insights ───────────────────────────────────────────────────
def recommendation_insights(df: pd.DataFrame | None = None) -> list[str]:
    """
    Generate specific, data-driven insights from the rules instead of
    two hardcoded generic strings.
    """
    rules = get_rules(df)
    insights = []

    if rules.empty:
        insights.append("No association rules found — try lowering min_support in apriori_model.py")
        return insights

    total_rules   = len(rules)
    strong_rules  = rules[rules["lift"] >= 2.0]
    avg_confidence = rules["confidence"].mean()
    avg_lift       = rules["lift"].mean()

    # Rule volume
    insights.append(f"{total_rules} product association rules discovered across your transactions")

    # Strong associations
    if not strong_rules.empty:
        top = strong_rules.iloc[0]
        ant = " + ".join(sorted(top["antecedents"]))
        con = " + ".join(sorted(top["consequents"]))
        insights.append(
            f"Strongest bundle: '{ant}' → '{con}' "
            f"(lift {top['lift']:.2f}x more likely than random)"
        )

    # Confidence level
    if avg_confidence >= 0.5:
        insights.append(
            f"High average confidence ({avg_confidence:.0%}) — recommendations are reliable"
        )
    else:
        insights.append(
            f"Average confidence is {avg_confidence:.0%} — consider bundling promotions to strengthen associations"
        )

    # Cross-sell potential
    if avg_lift > 1.5:
        insights.append(
            f"Average lift of {avg_lift:.2f}x indicates strong cross-selling potential — "
            "recommend pairing high-lift products at checkout"
        )

    # Products most often recommended
    all_consequents = []
    for fs in rules["consequents"]:
        all_consequents.extend(list(fs))

    if all_consequents:
        top_product = pd.Series(all_consequents).value_counts().idxmax()
        insights.append(
            f"'{top_product}' appears most frequently as a recommended product — "
            "consider featuring it in bundle deals"
        )

    return insights


# ── 6. Agent summary ──────────────────────────────────────────────────────────
def recommendation_agent_summary(df: pd.DataFrame | None = None) -> dict:
    """
    Master output function called by dashboard and main.py.

    Returns
    -------
    dict with keys:
      top_rules    — top 5 rules, display-ready DataFrame
      all_rules    — all rules with string columns (first 10 rows)
      insights     — list of data-driven insight strings
      rule_count   — total number of rules generated
    """
    rules    = get_rules(df)
    top      = top_recommendations(df, top_n=5)
    all_fmt  = formatted_rules(df)
    insights = recommendation_insights(df)

    return {
        "top_rules":  top,
        "all_rules":  all_fmt.head(10),
        "insights":   insights,
        "rule_count": len(rules),
    }