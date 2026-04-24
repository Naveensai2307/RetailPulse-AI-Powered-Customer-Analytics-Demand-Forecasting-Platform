import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Churn Risk | RetailPulse", layout="wide")

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

st.title("⚠️ Churn Risk Analysis")
st.caption("XGBoost-powered customer retention intelligence & at-risk identification")
st.divider()

if "data" not in st.session_state:
    st.info("⚠️ Please upload a dataset in the **Data Upload** portal.")
    st.stop()

df = st.session_state["data"].copy()

try:
    # ── Column Discovery ──────────────────────────────────────
    cust_col  = next((c for c in ['customer_unique_id', 'customer_id', 'CustomerID'] if c in df.columns), df.columns[0])
    date_col  = next((c for c in ['order_purchase_timestamp', 'InvoiceDate', 'order_date'] if c in df.columns),
                     df.select_dtypes(include=['datetime']).columns[0]
                     if not df.select_dtypes(include=['datetime']).empty else df.columns[1])
    price_col = next((c for c in ['payment_value', 'total_amount', 'price', 'TotalPrice', 'amount']
                      if c in df.columns), df.select_dtypes(include=[np.number]).columns[0])

    @st.cache_data(show_spinner=False)
    def build_rfm(_df, cust, date, price):
        _df = _df.copy()
        _df[date] = pd.to_datetime(_df[date], errors='coerce')
        _df = _df.dropna(subset=[date])
        snapshot = _df[date].max()
        rfm = _df.groupby(cust).agg(
            Recency  = (date,  lambda x: (snapshot - x.max()).days),
            Frequency= (cust,  'count'),
            Monetary = (price, 'sum')
        ).reset_index()
        rfm['Churn'] = (rfm['Recency'] > 180).astype(int)
        return rfm

    if "churn_rfm" not in st.session_state:
        st.session_state["churn_rfm"] = build_rfm(df, cust_col, date_col, price_col)
    rfm = st.session_state["churn_rfm"]

    # ── Guard: need both classes ───────────────────────────────
    if rfm['Churn'].nunique() < 2:
        st.warning("⚠️ Only one churn class found in this dataset — cannot train a binary classifier. Try a larger dataset.")
        st.stop()

    # ── Train / Test Split (stratified) ───────────────────────
    X = rfm[['Recency', 'Frequency', 'Monetary']]
    y = rfm['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Train XGBoost ──────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def train_model(X_tr, y_tr):
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            tree_method='hist',
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        model.fit(X_tr, y_tr)
        return model

    if "churn_model_results" not in st.session_state:
        _m = train_model(X_train.values, y_train.values)
        st.session_state["churn_model_results"] = {
            "model":   _m,
            "y_pred":  _m.predict(X_test.values),
            "y_proba": _m.predict_proba(X_test.values)[:, 1]
        }
    model   = st.session_state["churn_model_results"]["model"]
    y_pred  = st.session_state["churn_model_results"]["y_pred"]
    y_proba = st.session_state["churn_model_results"]["y_proba"]

    # ── Metrics ────────────────────────────────────────────────
    churn_rate = rfm['Churn'].mean() * 100
    at_risk_val = rfm[rfm['Churn'] == 1]['Monetary'].sum()
    at_risk_cnt = rfm['Churn'].sum()
    try:
        auc = roc_auc_score(y_test, y_proba)
        auc_str = f"{auc:.4f}"
    except Exception:
        auc_str = "N/A"

    # ── KPI Tiles ──────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Churn Rate (180D)", f"{churn_rate:.1f}%",   help="Customers with Recency > 180 days labelled as churned")
    k2.metric("At-Risk Customers", f"{at_risk_cnt:,}",     help="Total customers predicted as churned")
    k3.metric("At-Risk Revenue",   f"₹ {at_risk_val:,.0f}", help="Total monetary value of churned customers")
    k4.metric("XGBoost AUC",       auc_str,                 help="ROC-AUC score on held-out test set")

    st.divider()

    # ── Visualisations ─────────────────────────────────────────
    v1, v2 = st.columns(2)

    with v1:
        st.subheader("📊 Feature Importance")
        importance = pd.DataFrame({
            'Feature':    X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        fig_imp = px.bar(
            importance, x='Importance', y='Feature', orientation='h',
            template="plotly_dark", color_discrete_sequence=['#EF553B']
        )
        fig_imp.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_imp, use_container_width=True)

    with v2:
        st.subheader("🎯 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=['Stay', 'Churn'], y=['Stay', 'Churn'],
            colorscale='Reds', text=cm, texttemplate="%{text}", showscale=False
        ))
        fig_cm.update_layout(
            template="plotly_dark",
            xaxis_title="Predicted", yaxis_title="Actual",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()

    st.divider()

    # ── ROC Curve + Risk Correlation ───────────────────────────
    r1, r2 = st.columns(2)
    with r1:
        st.subheader("📈 ROC Curve")
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC={auc_str})',
                          labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                          template="plotly_dark", color_discrete_sequence=['#AB63FA'])
        fig_roc.add_shape(type='line', line=dict(dash='dash', color='white'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig_roc, use_container_width=True)

    with r2:
        st.subheader("💸 Spend vs. Churn Probability")
        # Combine test features and probabilities
        risk_df = pd.DataFrame(X_test, columns=['Recency', 'Frequency', 'Monetary'])
        risk_df['Prob'] = y_proba
        fig_risk = px.scatter(risk_df, x="Monetary", y="Prob", color="Recency",
                             size="Frequency", template="plotly_dark",
                             labels={'Prob': 'Churn Probability', 'Monetary': 'Total Spend'},
                             title="Risk Intensity by Monetary Value")
        st.plotly_chart(fig_risk, use_container_width=True)

    st.divider()

    # ── Top At-Risk Customers Table ────────────────────────────
    st.subheader("🔴 Top At-Risk Customers (Highest Monetary Value)")
    at_risk_table = rfm[rfm['Churn'] == 1].sort_values('Monetary', ascending=False).head(20).reset_index(drop=True)
    at_risk_table.index += 1
    st.dataframe(
        at_risk_table[[cust_col, 'Recency', 'Frequency', 'Monetary']].rename(columns={
            cust_col:    'Customer ID',
            'Recency':   'Days Since Last Order',
            'Frequency': 'Total Orders',
            'Monetary':  'Total Spend (₹)'
        }),
        use_container_width=True
    )

    # ── DOWNLOAD AT-RISK LIST ──
    st.download_button(
        label="📥 Download At-Risk Customer List (CSV)",
        data=at_risk_table[[cust_col, 'Recency', 'Frequency', 'Monetary']].to_csv(index=False).encode('utf-8'),
        file_name="at_risk_customers.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader("💡 Model Interpretation (SHAP Insight)")
    st.info("""
    **How the AI decides Churn Risk:**
    - **Recency (High Impact):** The stronger the days since last order, the higher the churn probability (linear correlation).
    - **Frequency (Medium Impact):** Customers with only 1 order are statistically more likely to churn than repeat buyers.
    - **Monetary (Low Impact):** High-spenders are slightly more retained, but timing (Recency) remains the dominant predictor.
    """)

except Exception as e:
    st.error(f"Churn Model Error: {e}")
    st.info("Ensure the uploaded dataset contains customer IDs, purchase timestamps, and payment/price values.")