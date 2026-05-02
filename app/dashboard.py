import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Supermarket AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0f1117; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2e3450;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-card .label {
        font-size: 13px;
        color: #8b93b0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        color: #e8eaf6;
    }
    .metric-card .delta {
        font-size: 12px;
        color: #66bb6a;
        margin-top: 4px;
    }

    /* Insight pills */
    .insight-pill {
        background: linear-gradient(90deg, #1a237e22, #1565c022);
        border-left: 3px solid #5c6bc0;
        border-radius: 0 8px 8px 0;
        padding: 10px 16px;
        margin: 6px 0;
        color: #c5cae9;
        font-size: 14px;
    }

    /* Section headers */
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #e8eaf6;
        border-bottom: 2px solid #3949ab;
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
        border-right: 1px solid #2e3450;
    }

    /* Page title */
    .page-title {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #7986cb, #42a5f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .page-subtitle {
        font-size: 14px;
        color: #8b93b0;
        margin-bottom: 24px;
    }

    /* Status badge */
    .badge-profit {
        background: #1b5e2044;
        color: #81c784;
        border: 1px solid #388e3c;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-loss {
        background: #b71c1c44;
        color: #ef9a9a;
        border: 1px solid #c62828;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-warning {
        background: #e65100aa;
        color: #ffcc80;
        border: 1px solid #ef6c00;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# IMPORTS (with error handling)
# ─────────────────────────────────────────────
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from utils.preprocessing import preprocess_data
    from models.lstm_pytorch import run_lstm
    from models.kmeans_model import label_clusters
    from agents.inventory_agent import inventory_agent_summary
    from agents.customer_agent import customer_agent_summary
    from agents.profit_agent import profit_agent_summary
    from agents.recommendation_agent import recommendation_agent_summary
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

# ─────────────────────────────────────────────
# CACHED DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    return preprocess_data()

@st.cache_data(show_spinner=False)
def get_inventory(df):
    return inventory_agent_summary(df)

@st.cache_data(show_spinner=False)
def get_profit(df):
    return profit_agent_summary(df)

@st.cache_data(show_spinner=False)
def get_customers(df):
    return customer_agent_summary(df)

@st.cache_data(show_spinner=False)
def get_clusters(df):
    return label_clusters(df)

@st.cache_data(show_spinner=False)
def get_recommendations():
    return recommendation_agent_summary()

# ─────────────────────────────────────────────
# HELPER COMPONENTS
# ─────────────────────────────────────────────
def metric_card(label, value, delta=None):
    delta_html = f'<div class="delta">▲ {delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def section_header(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def show_insights(insights):
    for insight in insights:
        st.markdown(f'<div class="insight-pill">💡 {insight}</div>', unsafe_allow_html=True)

def show_error(label, err):
    st.error(f"❌ {label} failed: {err}")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Loading data..."):
    try:
        df = load_data()
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        st.stop()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Supermarket AI")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Sales & Inventory", "👥 Customers & Recommendations", "🔮 Forecasting"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    # Quick stats in sidebar
    try:
        st.markdown("**📌 Quick Stats**")
        st.caption(f"📦 Products: **{df['Product_Name'].nunique()}**")
        st.caption(f"👥 Customers: **{df['Customer_ID'].nunique()}**")
        st.caption(f"📅 Date Range: **{df['Date'].min().date()} → {df['Date'].max().date()}**")
        total_p = df["Profit"].sum()
        badge = "badge-profit" if total_p > 0 else "badge-loss"
        label = "Profitable ✓" if total_p > 0 else "In Loss ✗"
        st.markdown(f'<span class="{badge}">{label}</span>', unsafe_allow_html=True)
    except Exception:
        pass

    st.markdown("---")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ═══════════════════════════════════════════
# PAGE 1 — SALES & INVENTORY
# ═══════════════════════════════════════════
if page == "📊 Sales & Inventory":

    st.markdown('<div class="page-title">📊 Sales & Inventory</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Track revenue, stock levels, and demand trends</div>', unsafe_allow_html=True)

    # ── Top KPI Row ──
    try:
        profit_data = get_profit(df)
        inventory = get_inventory(df)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            total = profit_data["total_profit"]
            metric_card("Total Profit", f"₹{total:,.0f}")
        with c2:
            metric_card("Top Products", str(len(profit_data["top_products"])))
        with c3:
            low_count = len(inventory["low_stock_products"])
            metric_card("Low Stock Items", str(low_count), "needs restock" if low_count > 0 else None)
        with c4:
            metric_card("Festival Demand Items", str(len(inventory["festival_top_products"])))
    except Exception as e:
        show_error("KPI Row", e)

    st.markdown("---")

    # ── Sales Trend ──
    section_header("📈 Daily Sales Trend")
    try:
        sales = df.groupby("Date")["Quantity_Sold"].sum().reset_index()
        fig = px.area(
            sales, x="Date", y="Quantity_Sold",
            color_discrete_sequence=["#5c6bc0"],
            template="plotly_dark",
            labels={"Quantity_Sold": "Units Sold", "Date": "Date"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified",
        )
        fig.update_traces(fill="tozeroy", fillcolor="rgba(92,107,192,0.15)", line_width=2)
        st.plotly_chart(fig, width='stretch')
    except Exception as e:
        show_error("Sales Trend", e)

    # ── Profit & Inventory side by side ──
    col_left, col_right = st.columns(2)

    with col_left:
        section_header("💰 Top Profitable Products")
        try:
            top = profit_data["top_products"].reset_index()
            top.columns = ["Product", "Profit"]
            fig2 = px.bar(
                top, x="Profit", y="Product", orientation="h",
                color="Profit", color_continuous_scale="Blues",
                template="plotly_dark",
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_showscale=False,
                yaxis=dict(categoryorder="total ascending"),
            )
            st.plotly_chart(fig2, width='stretch')

            # Loss products
            loss = profit_data.get("loss_products")
            if loss is not None and not loss.empty:
                st.markdown('<span class="badge-loss">⚠ Loss-Making Products Detected</span>', unsafe_allow_html=True)
                st.dataframe(loss.reset_index().rename(columns={"Product_Name": "Product", "Profit": "Loss (₹)"}),
                             width='stretch', hide_index=True)
        except Exception as e:
            show_error("Profit Analysis", e)

    with col_right:
        section_header("📦 Inventory Status")
        try:
            low_stock = inventory["low_stock_products"]
            restock = inventory["restock_suggestions"]

            if not low_stock.empty:
                st.markdown('<span class="badge-warning">🔴 Low Stock Alert</span>', unsafe_allow_html=True)
                st.markdown("")
                low_df = low_stock.reset_index()
                low_df.columns = ["Product", "Current Stock"]
                low_df["Restock Qty"] = low_df["Product"].map(restock).fillna(0).astype(int)
                st.dataframe(low_df, width='stretch', hide_index=True)
            else:
                st.success("✅ All products adequately stocked")

            section_header("🎉 Festival Top Products")
            fest = inventory["festival_top_products"].reset_index()
            fest.columns = ["Product", "Units Sold"]
            fig3 = px.bar(
                fest, x="Product", y="Units Sold",
                color="Units Sold", color_continuous_scale="Oranges",
                template="plotly_dark",
            )
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig3, width='stretch')
        except Exception as e:
            show_error("Inventory", e)

    # ── Insights ──
    section_header("🧠 AI Inventory Insights")
    try:
        show_insights(inventory["insights"])
    except Exception as e:
        show_error("Insights", e)


# ═══════════════════════════════════════════
# PAGE 2 — CUSTOMERS & RECOMMENDATIONS
# ═══════════════════════════════════════════
elif page == "👥 Customers & Recommendations":

    st.markdown('<div class="page-title">👥 Customers & Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Understand customer segments and drive cross-selling</div>', unsafe_allow_html=True)

    # ── Segment KPIs ──
    try:
        clusters = get_clusters(df)
        c1, c2, c3 = st.columns(3)
        for col, seg in zip([c1, c2, c3], ["High Value", "Medium Value", "Low Value"]):
            count = len(clusters[clusters["Segment"] == seg])
            with col:
                metric_card(f"{seg} Customers", str(count))
    except Exception as e:
        show_error("Segment KPIs", e)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        section_header("👥 Customer Segments")
        try:
            seg_counts = clusters["Segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]
            colors = {"High Value": "#66bb6a", "Medium Value": "#ffa726", "Low Value": "#ef5350"}
            fig = px.pie(
                seg_counts, names="Segment", values="Count",
                color="Segment", color_discrete_map=colors,
                template="plotly_dark", hole=0.45,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, width='stretch')
        except Exception as e:
            show_error("Segment Chart", e)

        section_header("📋 Customer Table")
        try:
            display_cols = [c for c in ["Customer_ID", "Quantity_Sold", "Profit", "Segment"] if c in clusters.columns]
            st.dataframe(
                clusters[display_cols].head(20),
                width='stretch',
                hide_index=True,
            )
        except Exception as e:
            show_error("Customer Table", e)

    with col_right:
        section_header("🎯 High vs Low Value Customers")
        try:
            customer = get_customers(df)

            tab1, tab2 = st.tabs(["⭐ High Value", "📉 Low Value"])
            with tab1:
                hv = customer["high_value_customers"]
                st.dataframe(hv, width='stretch', hide_index=True)
            with tab2:
                lv = customer["low_value_customers"]
                st.dataframe(lv, width='stretch', hide_index=True)
        except Exception as e:
            show_error("Customer Agent", e)

        section_header("🧠 Marketing Insights")
        try:
            show_insights(customer["insights"])
        except Exception as e:
            show_error("Marketing Insights", e)

    # ── Recommendations ──
    st.markdown("---")
    section_header("🛒 Product Recommendations (Association Rules)")
    try:
        rec = get_recommendations()
        top_rules = rec["top_rules"]

        if not top_rules.empty:
            c1, c2 = st.columns([3, 1])
            with c1:
                st.dataframe(
                    top_rules.style.background_gradient(subset=["Lift"], cmap="Blues"),
                    width='stretch',
                    hide_index=True,
                )
            with c2:
                st.markdown("**How to read this:**")
                st.caption("**Antecedents** → products the customer already has")
                st.caption("**Consequents** → products to recommend")
                st.caption("**Lift > 1** → genuine association (not random)")
                st.caption("Higher lift = stronger recommendation")
        else:
            st.info("No association rules generated yet. Try lowering min_support in the Apriori model.")

        show_insights(rec["insights"])
    except Exception as e:
        show_error("Recommendations", e)


# ═══════════════════════════════════════════
# PAGE 3 — FORECASTING
# ═══════════════════════════════════════════
elif page == "🔮 Forecasting":

    st.markdown('<div class="page-title">🔮 Sales Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">LSTM neural network predicts next-day total sales</div>', unsafe_allow_html=True)

    # ── Historical sales context ──
    section_header("📈 Recent Sales (Last 30 Days)")
    try:
        sales = df.groupby("Date")["Quantity_Sold"].sum().reset_index()
        recent = sales.tail(30)
        fig = px.line(
            recent, x="Date", y="Quantity_Sold",
            template="plotly_dark",
            markers=True,
            color_discrete_sequence=["#42a5f5"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, width='stretch')

        avg_7 = sales["Quantity_Sold"].tail(7).mean()
        avg_30 = sales["Quantity_Sold"].tail(30).mean()
        c1, c2 = st.columns(2)
        with c1:
            metric_card("7-Day Avg Sales", f"{avg_7:.0f} units")
        with c2:
            metric_card("30-Day Avg Sales", f"{avg_30:.0f} units")
    except Exception as e:
        show_error("Recent Sales", e)

    # ── LSTM Prediction ──
    st.markdown("---")
    section_header("🤖 LSTM Prediction")
    st.info("ℹ️ The model trains on your full sales history each time. This may take 10–30 seconds.")

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("▶ Run LSTM Prediction", type="primary", width='stretch'):
            with st.spinner("Training LSTM model..."):
                try:
                    start = time.time()
                    prediction = run_lstm(df)
                    elapsed = time.time() - start

                    st.session_state["lstm_result"] = prediction
                    st.session_state["lstm_time"] = elapsed
                except Exception as e:
                    st.error(f"❌ LSTM failed: {e}")

    with col2:
        if "lstm_result" in st.session_state:
            pred = st.session_state["lstm_result"]
            elapsed = st.session_state.get("lstm_time", 0)
            st.success(f"✅ Prediction complete in {elapsed:.1f}s")
            metric_card("Predicted Next-Day Sales", f"{pred:.0f} units")

            avg = df.groupby("Date")["Quantity_Sold"].sum().mean()
            diff = pred - avg
            direction = "above" if diff > 0 else "below"
            st.caption(f"📊 That's **{abs(diff):.0f} units {direction}** the daily average ({avg:.0f} units)")
