"""
╔══════════════════════════════════════════════════════╗
║         🛒  SUPERMARKET AI SYSTEM  —  main.py        ║
╚══════════════════════════════════════════════════════╝
Entry point for running the full AI pipeline from CLI.
Usage:  python main.py
        python main.py --skip-lstm      (faster, no training)
        python main.py --export         (save results to CSV)
"""

import sys
import time
import argparse
import traceback
from datetime import datetime

import pandas as pd

# ── Imports ──────────────────────────────────────────────────────────────────
from utils.preprocessing import preprocess_data, data_summary

from models.lstm_pytorch import run_lstm
from models.kmeans_model import label_clusters, cluster_summary
from models.apriori_model import get_rules

from agents.inventory_agent import inventory_agent_summary
from agents.customer_agent import customer_agent_summary
from agents.profit_agent import profit_agent_summary
from agents.recommendation_agent import recommendation_agent_summary


# ── Console helpers ───────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"

def header(title: str):
    width = 56
    print(f"\n{BOLD}{CYAN}{'─' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * width}{RESET}")

def ok(msg: str):
    print(f"  {GREEN}✔  {msg}{RESET}")

def warn(msg: str):
    print(f"  {YELLOW}⚠  {msg}{RESET}")

def fail(msg: str):
    print(f"  {RED}✘  {msg}{RESET}")

def info(msg: str):
    print(f"  {DIM}   {msg}{RESET}")

def kv(key: str, value):
    print(f"  {DIM}{key:<28}{RESET}{BOLD}{value}{RESET}")

def timer_label(seconds: float) -> str:
    return f"{seconds:.2f}s"


# ── Step runner ───────────────────────────────────────────────────────────────

def run_step(label: str, fn, *args, **kwargs):
    """
    Runs fn(*args, **kwargs), prints timing and result status.
    Returns (result, elapsed) or (None, elapsed) on failure.
    """
    start = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - start
        ok(f"{label}  {DIM}({timer_label(elapsed)}){RESET}")
        return result, elapsed
    except Exception as e:
        elapsed = time.time() - start
        fail(f"{label}  {DIM}({timer_label(elapsed)}){RESET}")
        info(f"Error: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return None, elapsed


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Supermarket AI — full pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-lstm",  action="store_true", help="Skip LSTM training (faster run)")
    parser.add_argument("--export",     action="store_true", help="Export results to CSV files in outputs/")
    parser.add_argument("--verbose",    action="store_true", help="Print full tracebacks on errors")
    return parser.parse_args()


# ── Export helper ─────────────────────────────────────────────────────────────

def export_results(results: dict):
    import os
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    exports = {
        "customer_segments":    results.get("clusters"),
        "cluster_summary":      results.get("cluster_sum"),
        "top_profitable":       results.get("profit", {}).get("top_products"),
        "loss_products":        results.get("profit", {}).get("loss_products"),
        "low_stock":            results.get("inventory", {}).get("low_stock_products"),
        "high_value_customers": results.get("customer", {}).get("high_value_customers"),
        "apriori_rules":        results.get("rules"),
        "top_recommendations":  results.get("rec", {}).get("top_rules"),
    }

    saved = []
    for name, data in exports.items():
        if data is None:
            continue
        if isinstance(data, pd.Series):
            data = data.reset_index()
        if isinstance(data, pd.DataFrame) and not data.empty:
            path = f"outputs/{name}_{timestamp}.csv"
            data.to_csv(path, index=False)
            saved.append(path)

    return saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    results = {}
    timings = {}
    errors  = []

    wall_start = time.time()

    print(f"\n{BOLD}{'═' * 56}{RESET}")
    print(f"{BOLD}  🛒  SUPERMARKET AI SYSTEM{RESET}")
    print(f"{BOLD}  {DIM}{datetime.now().strftime('%A, %d %B %Y  %H:%M:%S')}{RESET}")
    print(f"{BOLD}{'═' * 56}{RESET}")

    # ── Step 1: Data ─────────────────────────────────────────────────────────
    header("Step 1 · Data Loading & Preprocessing")
    df, t = run_step("Load & preprocess data", preprocess_data)
    timings["preprocessing"] = t

    if df is None:
        fail("Cannot continue without data. Exiting.")
        sys.exit(1)

    summary = data_summary(df)
    kv("Rows loaded",        f"{summary['Total Rows']:,}")
    kv("Unique products",    summary["Total Products"])
    kv("Unique customers",   summary["Total Customers"])
    kv("Total profit (raw)", f"₹{summary['Total Profit']:,.2f}")

    # ── Step 2: LSTM ─────────────────────────────────────────────────────────
    header("Step 2 · Sales Forecasting (LSTM)")
    if args.skip_lstm:
        warn("Skipped (--skip-lstm flag set)")
    else:
        prediction, t = run_step("Train & predict next-day sales", run_lstm, df)
        timings["lstm"] = t
        if prediction is not None:
            avg_sales = df.groupby("Date")["Quantity_Sold"].sum().mean()
            diff      = prediction - avg_sales
            direction = "above" if diff >= 0 else "below"
            kv("Predicted next-day sales", f"{prediction:.2f} units")
            kv("vs daily average",         f"{diff:+.2f} ({direction} avg of {avg_sales:.1f})")
            results["lstm_prediction"] = prediction
        else:
            errors.append("LSTM")

    # ── Step 3: K-Means ───────────────────────────────────────────────────────
    header("Step 3 · Customer Segmentation (K-Means)")
    clusters, t = run_step("Cluster customers", label_clusters, df)
    timings["kmeans"] = t

    if clusters is not None:
        results["clusters"] = clusters
        for seg in ["High Value", "Medium Value", "Low Value"]:
            count = len(clusters[clusters["Segment"] == seg]) if "Segment" in clusters.columns else "?"
            kv(f"  {seg}", count)

        clus_sum, _ = run_step("Cluster summary", cluster_summary, df)
        results["cluster_sum"] = clus_sum
        if clus_sum is not None:
            print()
            print(clus_sum.to_string(index=False))
    else:
        errors.append("K-Means")

    # ── Step 4: Apriori ───────────────────────────────────────────────────────
    header("Step 4 · Product Recommendations (Apriori)")
    rules, t = run_step("Generate association rules", get_rules)
    timings["apriori"] = t

    if rules is not None:
        if rules.empty:
            warn("No rules generated — try lowering min_support in apriori_model.py")
        else:
            results["rules"] = rules
            kv("Rules generated", len(rules))
            display = rules.copy()
            display["antecedents"] = display["antecedents"].apply(lambda x: ", ".join(list(x)))
            display["consequents"] = display["consequents"].apply(lambda x: ", ".join(list(x)))
            print()
            print(display[["antecedents", "consequents", "lift"]].head(5).to_string(index=False))
    else:
        errors.append("Apriori")

    # ── Step 5: Inventory Agent ───────────────────────────────────────────────
    header("Step 5 · Inventory Agent")
    inventory, t = run_step("Run inventory agent", inventory_agent_summary, df)
    timings["inventory_agent"] = t

    if inventory is not None:
        results["inventory"] = inventory
        kv("Low-stock products",    len(inventory["low_stock_products"]))
        kv("Festival top products", len(inventory["festival_top_products"]))
        print()
        print("  Top Demand Products:")
        print(inventory["top_demand_products"].to_string())
        print()
        for insight in inventory["insights"]:
            info(f"💡 {insight}")
    else:
        errors.append("Inventory Agent")

    # ── Step 6: Customer Agent ────────────────────────────────────────────────
    header("Step 6 · Customer Agent")
    customer, t = run_step("Run customer agent", customer_agent_summary, df)
    timings["customer_agent"] = t

    if customer is not None:
        results["customer"] = customer
        print()
        print("  High Value Customers (sample):")
        print(customer["high_value_customers"].to_string(index=False))
        print()
        for insight in customer["insights"]:
            info(f"💡 {insight}")
    else:
        errors.append("Customer Agent")

    # ── Step 7: Profit Agent ──────────────────────────────────────────────────
    header("Step 7 · Profit Agent")
    profit, t = run_step("Run profit agent", profit_agent_summary, df)
    timings["profit_agent"] = t

    if profit is not None:
        results["profit"] = profit
        total  = profit["total_profit"]
        status = f"{GREEN}PROFIT{RESET}" if total > 0 else f"{RED}LOSS{RESET}"
        kv("Store status",  status)
        kv("Total profit",  f"₹{total:,.2f}")
        print()
        print("  Top Profitable Products:")
        print(profit["top_products"].to_string())
        if not profit["loss_products"].empty:
            print()
            warn("Loss-making products:")
            print(profit["loss_products"].to_string())
        print()
        for insight in profit["insights"]:
            info(f"💡 {insight}")
    else:
        errors.append("Profit Agent")

    # ── Step 8: Recommendation Agent ─────────────────────────────────────────
    header("Step 8 · Recommendation Agent")
    rec, t = run_step("Run recommendation agent", recommendation_agent_summary)
    timings["recommendation_agent"] = t

    if rec is not None:
        results["rec"] = rec
        print()
        print("  Top Recommendation Rules:")
        print(rec["top_rules"].to_string(index=False))
        print()
        for insight in rec["insights"]:
            info(f"💡 {insight}")
    else:
        errors.append("Recommendation Agent")

    # ── Export ────────────────────────────────────────────────────────────────
    if args.export:
        header("Export")
        saved = export_results(results)
        for path in saved:
            ok(f"Saved → {path}")
        if not saved:
            warn("Nothing exported (all results empty or failed)")

    # ── Final Summary ─────────────────────────────────────────────────────────
    wall_elapsed = time.time() - wall_start

    print(f"\n{BOLD}{'═' * 56}{RESET}")
    print(f"{BOLD}  PIPELINE SUMMARY{RESET}")
    print(f"{'─' * 56}")

    total_steps  = 8 if not args.skip_lstm else 7
    failed_steps = len(errors)
    passed_steps = total_steps - failed_steps

    kv("Steps passed",   f"{GREEN}{passed_steps}/{total_steps}{RESET}")
    if errors:
        kv("Steps failed",  f"{RED}{', '.join(errors)}{RESET}")
    kv("Total wall time", timer_label(wall_elapsed))

    print(f"\n  {'─' * 40}")
    print(f"  {'Step':<30} {'Time':>8}")
    print(f"  {'─' * 40}")
    for step, secs in timings.items():
        bar = "█" * min(int(secs / wall_elapsed * 20), 20)
        print(f"  {step:<30} {timer_label(secs):>8}  {DIM}{bar}{RESET}")
    print(f"  {'─' * 40}")

    if failed_steps == 0:
        print(f"\n{BOLD}{GREEN}  🎉  All steps completed successfully!{RESET}\n")
    else:
        print(f"\n{BOLD}{YELLOW}  ⚠   Completed with {failed_steps} error(s). Use --verbose for details.{RESET}\n")

    return 0 if failed_steps == 0 else 1


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.exit(main())