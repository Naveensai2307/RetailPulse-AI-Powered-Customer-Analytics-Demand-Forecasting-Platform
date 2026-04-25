import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prometheus_client import start_http_server, Counter, Gauge
import threading

# -------------------------------------------------
# MONITORING SETUP
# -------------------------------------------------
@st.cache_resource
def start_metrics_server():
    try:
        start_http_server(8001)
        print("Prometheus metrics server started on port 8001")
    except Exception as e:
        print(f"Metrics server error: {e}")

# Start the server once
start_metrics_server()

# Define some basic metrics (Cached to prevent duplication errors)
@st.cache_resource
def get_request_counter():
    return Counter('retailpulse_requests_total', 'Total number of dashboard visits')

REQUEST_COUNT = get_request_counter()
REQUEST_COUNT.inc() # Increment on load

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="RetailPulse | AI Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    [data-testid="stSidebarNav"], [data-testid="stSidebarNavItems"],
    .st-emotion-cache-16idsys p { display: none !important; }
    .stApp { opacity: 1 !important; transition: none !important; animation: none !important; }
    div[data-testid="stStatusWidget"], div[data-testid="stStatusWidget"] * { visibility: hidden !important; display: none !important; }
    section[data-testid="stSidebar"] .block-container { padding-top: 0rem !important; }
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

st.title("📊 RetailPulse – AI-Powered Customer Analytics & Demand Forecasting Platform")
st.caption("Upload your retail dataset to unlock all analytical modules")
st.divider()

# -------------------------------------------------
# COLUMN MAPPING
# -------------------------------------------------
def map_columns(df):
    mapping = {
        'order_purchase_timestamp': 'InvoiceDate',
        'customer_id': 'CustomerID',
        'total_amount': 'TotalPrice',
        'payment_value': 'TotalPrice',
        'price': 'UnitPrice',
        'order_id': 'InvoiceNo'
    }
    for old, new in mapping.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    return df

# -------------------------------------------------
# PRE-COMPUTATION — runs once on upload
# -------------------------------------------------
def precompute_all(df):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    # Column discovery
    cust_col  = next((c for c in ['customer_unique_id', 'customer_id', 'CustomerID'] if c in df.columns), df.columns[0])
    date_col  = next((c for c in ['InvoiceDate', 'order_purchase_timestamp', 'order_date'] if c in df.columns),
                     df.select_dtypes(include=['datetime']).columns[0]
                     if not df.select_dtypes(include=['datetime']).empty else df.columns[1])
    price_col = next((c for c in ['payment_value', 'TotalPrice', 'total_amount', 'UnitPrice', 'price']
                      if c in df.columns), df.select_dtypes(include=[np.number]).columns[0])
    prod_col  = next((c for c in ['product_id', 'StockCode', 'product_name'] if c in df.columns), None)
    qty_col   = next((c for c in ['order_id', 'InvoiceNo', 'order_item_id'] if c in df.columns), None)

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
    d = d.dropna(subset=[date_col])
    snapshot = d[date_col].max()

    # 1. RFM + K-Means (Segmentation)
    rfm = d.groupby(cust_col).agg(
        Recency  =(date_col,  lambda x: (snapshot - x.max()).days),
        Frequency=(cust_col,  'count'),
        Monetary =(price_col, 'sum')
    ).reset_index()
    scaler     = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    kmeans     = KMeans(n_clusters=6, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    rfm['Persona'] = rfm['Cluster'].map({
        0: "Champions", 1: "At-Risk", 2: "New Customers",
        3: "Loyalists",  4: "Hibernating", 5: "Promising"
    })
    st.session_state['rfm_segments'] = rfm

    # 2. Churn RFM
    churn_rfm = rfm[[cust_col, 'Recency', 'Frequency', 'Monetary']].copy()
    churn_rfm['Churn'] = (churn_rfm['Recency'] > 180).astype(int)
    st.session_state['churn_rfm'] = churn_rfm

    # 3. XGBoost Churn Model
    if churn_rfm['Churn'].nunique() == 2:
        X = churn_rfm[['Recency', 'Frequency', 'Monetary']]
        y = churn_rfm['Churn']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, tree_method='hist', random_state=42,
            eval_metric='logloss', verbosity=0
        )
        model.fit(X_train.values, y_train.values)
        st.session_state['churn_model_results'] = {
            "model":   model,
            "y_pred":  model.predict(X_test.values),
            "y_proba": model.predict_proba(X_test.values)[:, 1],
            "X_test":  X_test,
            "y_test":  y_test
        }

    # 4. Inventory stats
    if prod_col and qty_col:
        inv = d.groupby(prod_col).agg(
            total_sold   =(qty_col,   'count'),
            total_revenue=(price_col, 'sum')
        ).reset_index()
        inv['avg_daily_sales'] = inv['total_sold'] / 730
        inv['reorder_point']   = (inv['avg_daily_sales'] * 7) + 5
        inv = inv.sort_values('total_sold', ascending=False).reset_index(drop=True)
        st.session_state['inventory_stats'] = (inv, prod_col, price_col)

    # 5. Prophet Demand Forecast
    try:
        from prophet import Prophet
        daily_sales = d.groupby(d[date_col].dt.date)[price_col].sum().reset_index()
        daily_sales.columns = ['ds', 'y']
        daily_sales = daily_sales.dropna()
        daily_sales['ds'] = pd.to_datetime(daily_sales['ds'])
        prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        prophet_model.fit(daily_sales)
        future   = prophet_model.make_future_dataframe(periods=90)
        forecast = prophet_model.predict(future)
        st.session_state['prophet_forecast']    = forecast
        st.session_state['prophet_daily_sales'] = daily_sales
        st.session_state['prophet_date_col']    = date_col
        st.session_state['prophet_price_col']   = price_col
    except Exception:
        pass  # Prophet optional — page falls back to fresh compute if unavailable

# -------------------------------------------------
# DATA UPLOAD
# -------------------------------------------------
st.header("Step 1: Data Ingestion")
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    df_new = (pd.read_csv(uploaded_file)
              if uploaded_file.name.endswith("csv")
              else pd.read_excel(uploaded_file))
    df_new = map_columns(df_new)
    st.session_state["data"] = df_new

    # Clear any stale precomputed data from a previous upload
    for key in ['rfm_segments', 'churn_rfm', 'churn_model_results', 'inventory_stats', 'sales_csv']:
        st.session_state.pop(key, None)

    with st.status("⚙️ Preparing all modules for instant access…", expanded=True) as status:
        st.write("📊 Computing RFM segmentation & K-Means clusters…")
        st.write("⚠️  Training XGBoost churn model…")
        st.write("📦 Calculating inventory reorder points…")
        st.write("📈 Training Prophet demand forecast (90-day)…")
        precompute_all(df_new)
        status.update(label="✅ All modules ready — navigate to any page instantly!", state="complete", expanded=False)

if "data" not in st.session_state:
    st.info("👋 Welcome! Please upload a dataset to begin.")
    st.stop()

# -------------------------------------------------
# INSTANT OVERVIEW
# -------------------------------------------------
df = st.session_state["data"]
st.divider()
st.header("📈 Instant Overview")
col1, col2, col3, col4 = st.columns(4)

revenue_col = 'TotalPrice' if 'TotalPrice' in df.columns else 'price'
col1.metric("Gross Revenue",   f"₹ {df[revenue_col].sum():,.0f}" if revenue_col in df.columns else "N/A")
col2.metric("Orders",          f"{len(df):,}")
col3.metric("Avg Order Value", f"₹ {df[revenue_col].mean():,.2f}" if revenue_col in df.columns else "N/A")
col4.metric("Unique Cities",   f"{df['customer_city'].nunique() if 'customer_city' in df.columns else 'N/A'}")

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)