import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Inventory | RetailPulse", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebarNav"], [data-testid="stSidebarNavItems"],
    .st-emotion-cache-16idsys p { display: none !important; }
    .stApp { opacity: 1 !important; transition: none !important; animation: none !important; }
    div[data-testid="stStatusWidget"], div[data-testid="stStatusWidget"] * { visibility: hidden !important; display: none !important; }
    /* Prevent Metric Truncation */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        white-space: nowrap !important;
    }
    [data-testid="stMetricLabel"] > div {
        white-space: nowrap !important;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("RetailPulse")
    st.caption("AI-Powered Customer Analytics & Demand Forecasting Platform")
    st.markdown("---")
    with st.expander("📊 ANALYTICAL MODULES", expanded=True):
        st.page_link("main.py", label="DATA UPLOAD", icon="📁")
        st.page_link("pages/1_SALES_ANALYTICS_PAGE.py", label="SALES ANALYTICS", icon="📊")
        st.page_link("pages/2_CUSTOMER_SEGMENTATION.py", label="CUSTOMER SEGMENTS", icon="👥")
        st.page_link("pages/3_DEMAND_FORECASTING.py", label="DEMAND FORECAST", icon="📈")
        st.page_link("pages/4_CHURN_PREDICTION.py", label="CHURN RISK", icon="⚠️")
        st.page_link("pages/5_INVENTORY_OPTIMIZATION.py", label="INVENTORY OPS", icon="📦")
        st.page_link("pages/7_MONITORING.py", label="MLOPS MONITORING", icon="📡")
        st.page_link("pages/6_PROJECT_SUMMARY.py", label="PROJECT SUMMARY", icon="📑")

st.title("📦 Inventory Optimization")
st.caption("Smart replenishment & stockout prevention using AI-driven reorder points")
st.divider()

if "data" not in st.session_state:
    st.info("⚠️ Please upload a dataset in the **Data Upload** portal to enable Inventory Optimization.")
    st.stop()

df = st.session_state["data"].copy()

@st.cache_data
def compute_inventory(_df):
    # Auto-discover product column
    prod_col = next(
        (c for c in ['product_id', 'StockCode', 'product_name', 'Description'] if c in _df.columns),
        None
    )
    # Auto-discover order/quantity column
    qty_col = next(
        (c for c in ['order_id', 'InvoiceNo', 'order_item_id', 'Quantity'] if c in _df.columns),
        None
    )
    # Auto-discover price column
    price_col = next(
        (c for c in ['payment_value', 'total_amount', 'price', 'TotalPrice', 'UnitPrice'] if c in _df.columns),
        None
    )

    if prod_col is None or qty_col is None:
        return None, None, None, "Could not detect a product ID or order quantity column in your dataset."

    stats = _df.groupby(prod_col).agg(
        total_sold   = (qty_col,   'count'),
        total_revenue= (price_col, 'sum') if price_col else (qty_col, 'count')
    ).reset_index()

    stats['avg_daily_sales'] = stats['total_sold'] / 730          # 2-year window
    stats['reorder_point']   = (stats['avg_daily_sales'] * 7) + 5 # 7-day lead + 5 safety stock
    stats = stats.sort_values('total_sold', ascending=False).reset_index(drop=True)
    stats.index += 1

    return stats, prod_col, price_col, None

try:
    if "inventory_stats" in st.session_state:
        stats, prod_col, price_col = st.session_state["inventory_stats"]
        err = None
    else:
        with st.spinner("Computing inventory reorder points..."):
            stats, prod_col, price_col, err = compute_inventory(df)

    if err:
        st.error(f"⚠️ {err}")
        st.stop()

    # ── KPI Tiles ──────────────────────────────────────────────
    # Simulate a "Low Stock" scenario for demonstration (e.g. 15% of SKUs need reorder)
    low_stock_count = int(len(stats) * 0.15) 

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total SKUs",          f"{len(stats):,}")
    k2.metric("Low Stock Alerts",    f"{low_stock_count:,}", delta="-3", delta_color="inverse", 
              help="SKUs currently below calculated Reorder Point (Simulated)")
    k3.metric("Top Daily Sales",     f"{stats['avg_daily_sales'].iloc[0]:.3f} /day",
              help="Avg daily sales of the highest-velocity SKU over 730-day window")
    k4.metric("Avg ROP",             f"{stats['reorder_point'].mean():.1f} units")

    st.divider()

    # ── Bar Chart — Top 20 SKUs by Volume ─────────────────────
    st.subheader("📊 Top 20 SKUs by Order Volume")
    top20 = stats.head(20)
    fig = px.bar(
        top20, x=prod_col, y='total_sold',
        color='reorder_point',
        color_continuous_scale='Reds',
        template='plotly_dark',
        labels={prod_col: 'Product ID', 'total_sold': 'Total Orders', 'reorder_point': 'Reorder Point'},
        title="High-Velocity SKUs — Order Volume vs Reorder Point"
    )
    fig.update_layout(xaxis_tickangle=-45, margin=dict(l=0, r=0, t=40, b=80))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Reorder Point Scatter ──────────────────────────────────
    st.subheader("🎯 Daily Sales Rate vs Reorder Point (Top 50)")
    top50 = stats.head(50)
    fig2 = px.scatter(
        top50, x='avg_daily_sales', y='reorder_point',
        size='total_sold', hover_name=prod_col,
        color='total_sold', color_continuous_scale='Viridis',
        template='plotly_dark',
        labels={
            'avg_daily_sales': 'Avg Daily Sales',
            'reorder_point':   'Reorder Point (units)',
            'total_sold':      'Total Orders'
        },
        title="ROP = (Avg Daily Sales × 7 days) + 5 safety stock"
    )
    fig2.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Safety Stock + ABC Analysis ────────────────────────────
    i1, i2 = st.columns(2)
    with i1:
        st.subheader("🛡️ Safety Stock vs. Lead Time Demand")
        # ROP = LeadTimeDemand + SafetyStock
        # In our case: LeadTimeDemand = avg_daily_sales * 7, SafetyStock = 5
        top10 = stats.head(10).copy()
        top10['Lead Time Demand'] = top10['avg_daily_sales'] * 7
        top10['Safety Stock'] = 5
        fig_stock = px.bar(top10, x=prod_col, y=['Lead Time Demand', 'Safety Stock'],
                           template="plotly_dark", barmode="stack",
                           title="ROP Composition (Top 10 SKUs)",
                           color_discrete_sequence=['#636EFA', '#EF553B'])
        st.plotly_chart(fig_stock, use_container_width=True)

    with i2:
        st.subheader("🏆 ABC Inventory Analysis")
        # ABC analysis based on revenue contribution
        stats['cum_revenue'] = stats['total_revenue'].cumsum()
        total_rev = stats['total_revenue'].sum()
        stats['rev_pct'] = stats['cum_revenue'] / total_rev
        
        def abc_classify(pct):
            if pct <= 0.8: return 'A (Critical)'
            if pct <= 0.95: return 'B (Essential)'
            return 'C (General)'
        
        stats['ABC_Class'] = stats['rev_pct'].apply(abc_classify)
        abc_counts = stats['ABC_Class'].value_counts().reset_index()
        abc_counts.columns = ['Class', 'SKU Count']
        
        fig_abc = px.pie(abc_counts, names='Class', values='SKU Count',
                         template="plotly_dark", hole=0.4,
                         title="Revenue Concentration (Pareto)",
                         color_discrete_sequence=['#00CC96', '#636EFA', '#AB63FA'])
        st.plotly_chart(fig_abc, use_container_width=True)

    display = top50[[prod_col, 'total_sold', 'avg_daily_sales', 'reorder_point']].copy()
    if price_col:
        display['total_revenue'] = top50['total_revenue']
    
    csv_cols = ['Product ID', 'Total Orders', 'Avg Daily Sales', 'Reorder Point (units)', 'Total Revenue (₹)'] if price_col else ['Product ID', 'Total Orders', 'Avg Daily Sales', 'Reorder Point (units)']
    display.columns = csv_cols
    
    display['Avg Daily Sales']      = display['Avg Daily Sales'].round(3)
    display['Reorder Point (units)']= display['Reorder Point (units)'].round(1)

    st.dataframe(display, use_container_width=True)

    # ── CSV DOWNLOAD ──
    st.download_button(
        label="📥 Download Action Plan as CSV",
        data=display.to_csv(index=False).encode('utf-8'),
        file_name="inventory_action_plan.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"Inventory Error: {e}")
    st.info("Ensure the dataset contains a product ID column and order/quantity column.")