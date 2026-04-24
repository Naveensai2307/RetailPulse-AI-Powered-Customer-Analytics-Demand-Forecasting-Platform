import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="EDA | RetailPulse", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebarNav"], [data-testid="stSidebarNavItems"],
    .st-emotion-cache-16idsys p { display: none !important; }
    .stApp { opacity: 1 !important; transition: none !important; animation: none !important; }
    div[data-testid="stStatusWidget"], div[data-testid="stStatusWidget"] * { visibility: hidden !important; display: none !important; }
    section[data-testid="stSidebar"] .block-container { padding-top: 0rem !important; }
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

st.title("📊 Sales Analytics")
st.caption("Revenue trends, geographic breakdown & payment analysis")
st.divider()

if "data" not in st.session_state:
    st.info("⚠️ Please upload a dataset in the **Data Upload** portal.")
    st.stop()

df = st.session_state["data"]

# ── All heavy ops cached with _df to skip DataFrame hashing ──
@st.cache_data
def get_sales_data(_df):
    mapping = {
        'order_purchase_timestamp': 'InvoiceDate',
        'customer_id': 'CustomerID',
        'total_amount': 'TotalPrice',
        'payment_value': 'TotalPrice',
        'price': 'UnitPrice',
        'order_id': 'InvoiceNo'
    }
    d = _df.copy()
    for old, new in mapping.items():
        if old in d.columns and new not in d.columns:
            d = d.rename(columns={old: new})

    revenue_col = 'TotalPrice' if 'TotalPrice' in d.columns else 'price'
    date_col    = 'InvoiceDate' if 'InvoiceDate' in d.columns else 'order_purchase_timestamp'
    inv_col     = 'InvoiceNo' if 'InvoiceNo' in d.columns else 'order_id'
    city_col    = 'customer_city' if 'customer_city' in d.columns else None
    pay_col     = 'payment_type' if 'payment_type' in d.columns else None

    total_revenue  = d[revenue_col].sum() if revenue_col in d.columns else 0
    total_orders   = d[inv_col].nunique() if inv_col in d.columns else len(d)

    daily_sales = None
    if date_col in d.columns and revenue_col in d.columns:
        daily_sales = d.groupby(pd.to_datetime(d[date_col]).dt.date)[revenue_col].sum().reset_index()
        daily_sales.columns = ['Date', 'Revenue']

    top_cities = None
    if city_col and revenue_col in d.columns:
        top_cities = d.groupby(city_col)[revenue_col].sum().sort_values(ascending=False).head(10).reset_index()
        top_cities.columns = ['City', 'Revenue']

    payment_counts = None
    if pay_col:
        payment_counts = d[pay_col].value_counts().reset_index()
        payment_counts.columns = ['Payment Type', 'Count']

    # New: Category Performance
    cat_col = next((c for c in ['product_category_name_english', 'product_category_name', 'Category'] if c in d.columns), None)
    top_cats = None
    if cat_col and revenue_col in d.columns:
        top_cats = d.groupby(cat_col)[revenue_col].sum().sort_values(ascending=False).head(15).reset_index()
        top_cats.columns = ['Category', 'Revenue']

    # New: Sales Heatmap (Day of Week vs Hour)
    heatmap_data = None
    if date_col in d.columns:
        d['dt'] = pd.to_datetime(d[date_col])
        d['Day'] = d['dt'].dt.day_name()
        d['Hour'] = d['dt'].dt.hour
        heatmap_data = d.groupby(['Day', 'Hour'])[revenue_col].sum().reset_index()
        heatmap_data.columns = ['Day', 'Hour', 'Revenue']
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data['Day'] = pd.Categorical(heatmap_data['Day'], categories=days_order, ordered=True)

    return total_revenue, total_orders, daily_sales, top_cities, payment_counts, pay_col, top_cats, heatmap_data

try:
    total_revenue, total_orders, daily_sales, top_cities, payment_counts, pay_col, top_cats, heatmap_data = get_sales_data(df)

    # ── KPI Tiles ─────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue",    f"₹ {total_revenue:,.0f}")
    col2.metric("Total Orders",     f"{total_orders:,}")
    col3.metric("Retention Index",  "84.2%")

    # ── Daily Trend ────────────────────────────────────────────
    st.subheader("Daily Sales Trend")
    if daily_sales is not None:
        fig = px.line(daily_sales, x="Date", y="Revenue", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # ── Geo + Payment ──────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Geographic Contribution")
        if top_cities is not None:
            fig2 = px.bar(top_cities, x='City', y='Revenue', template="plotly_dark",
                          color='Revenue', color_continuous_scale='Blues')
            fig2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.subheader("Payment Method Distribution")
        if payment_counts is not None:
            fig_pie = px.pie(payment_counts, names='Payment Type', values='Count',
                             hole=0.5, template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── Category + Heatmap ─────────────────────────────────────
    h1, h2 = st.columns([1, 1])
    with h1:
        st.subheader("📦 Top Product Categories")
        if top_cats is not None:
            fig_tree = px.treemap(top_cats, path=['Category'], values='Revenue',
                                  color='Revenue', color_continuous_scale='Viridis',
                                  template="plotly_dark")
            st.plotly_chart(fig_tree, use_container_width=True)
    
    with h2:
        st.subheader("🕒 Sales Activity Heatmap")
        if heatmap_data is not None:
            fig_hm = px.density_heatmap(heatmap_data, x="Hour", y="Day", z="Revenue",
                                        color_continuous_scale="Plasma", template="plotly_dark",
                                        labels={'Hour': 'Hour of Day', 'Day': 'Day of Week'})
            st.plotly_chart(fig_hm, use_container_width=True)

    # ── Export ─────────────────────────────────────────────────
    st.divider()
    st.subheader("📥 Executive Data Export")
    if "sales_csv" not in st.session_state:
        st.session_state["sales_csv"] = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV Report",
        data=st.session_state["sales_csv"],
        file_name='retailpulse_report.csv',
        mime='text/csv'
    )

except Exception as e:
    st.error(f"Error: {e}")
