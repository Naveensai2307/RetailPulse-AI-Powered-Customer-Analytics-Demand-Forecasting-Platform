import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Segmentation | RetailPulse", layout="wide")

# --- CSS: Minimal Overrides ---
st.markdown("""
    <style>
    /* Aggressive Sidebar Lockdown */
    [data-testid="stSidebarNav"], 
    [data-testid="stSidebarNavItems"],
    .st-emotion-cache-16idsys p { 
        display: none !important; 
    }
    
    /* Zero-Flicker & Instant Transition Logic */
    .stApp { 
        opacity: 1 !important;
        transition: none !important;
        animation: none !important;
    }
    
    /* Hide all loading overlays strictly */
    div[data-testid="stStatusWidget"],
    div[data-testid="stStatusWidget"] * { 
        visibility: hidden !important; 
        display: none !important;
    }
    
    /* Adjust sidebar padding */
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

# --- CUSTOM SIDEBAR NAVIGATION ---
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

st.title("👥 Customer Segmentation")
st.caption("RFM analysis & K-Means clustering to identify Champions, Loyalists, At-Risk & New Customers")
st.divider()

if "data" not in st.session_state:
    st.info("⚠️ System Standby: Please upload a dataset in the **Data Upload** portal.")
    st.stop()

df = st.session_state["data"]

@st.cache_data(show_spinner=False)
def run_segmentation(_df):
    # Auto-discover columns
    cust_col = next((c for c in ['customer_unique_id', 'customer_id', 'CustomerID'] if c in _df.columns), _df.columns[0])
    date_col = next((c for c in ['order_purchase_timestamp', 'InvoiceDate', 'order_date'] if c in _df.columns), _df.select_dtypes(include=['datetime']).columns[0] if not _df.select_dtypes(include=['datetime']).empty else _df.columns[1])
    price_keywords = ['payment_value', 'price', 'total_amount', 'amount', 'monetary', 'TotalPrice']
    price_col = next((c for c in price_keywords if c in _df.columns), _df.select_dtypes(include=[np.number]).columns[0])

    _df = _df.copy()
    _df[date_col] = pd.to_datetime(_df[date_col])
    snapshot_date = _df[date_col].max()

    rfm = _df.groupby(cust_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,
        cust_col: 'count',
        price_col: 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    persona_map = {0: "Champions", 1: "At-Risk", 2: "New Customers", 3: "Loyalists", 4: "Hibernating", 5: "Promising"}
    rfm['Persona'] = rfm['Cluster'].map(persona_map)
    return rfm

try:
    if "rfm_segments" not in st.session_state:
        st.session_state["rfm_segments"] = run_segmentation(df)
    rfm = st.session_state["rfm_segments"]

    # --- METRICS ---
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Champions", len(rfm[rfm['Cluster']==0]))
    m2.metric("At-Risk", len(rfm[rfm['Cluster']==1]))
    m3.metric("New Customers", len(rfm[rfm['Cluster']==2]))
    m4.metric("Loyalists", len(rfm[rfm['Cluster']==3]))
    m5.metric("Hibernating", len(rfm[rfm['Cluster']==4]))
    m6.metric("Promising", len(rfm[rfm['Cluster']==5]))

    # --- VISUALIZATIONS ---
    st.divider()
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("🥧 Market Share by Segment")
        fig_pie = px.pie(rfm, names='Persona', values='Monetary', hole=0.4, template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("📐 Segment Comparison (Avg Monetary)")
        avg_monetary = rfm.groupby('Persona')['Monetary'].mean().reset_index()
        fig_bar = px.bar(avg_monetary, x='Persona', y='Monetary', template="plotly_dark", color='Persona')
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── 3D Clusters + RFM Box Plot ─────────────────────────────
    d1, d2 = st.columns(2)
    with d1:
        st.subheader("🌌 3D RFM Cluster Map")
        # Sample data if too large for 3D performance
        sample_rfm = rfm.sample(min(2000, len(rfm))) if len(rfm) > 2000 else rfm
        fig_3d = px.scatter_3d(sample_rfm, x='Recency', y='Frequency', z='Monetary',
                               color='Persona', opacity=0.7, template="plotly_dark",
                               title="Spatial distribution of clusters")
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig_3d, use_container_width=True)

    with d2:
        st.subheader("📦 RFM Metric Distribution")
        metric_choice = st.selectbox("Choose Metric to Analyze", ["Recency", "Frequency", "Monetary"])
        fig_box = px.box(rfm, x="Persona", y=metric_choice, color="Persona",
                         template="plotly_dark", notched=True)
        st.plotly_chart(fig_box, use_container_width=True)

except Exception as e:
    st.error(f"Segmentation Engine Error: {e}")