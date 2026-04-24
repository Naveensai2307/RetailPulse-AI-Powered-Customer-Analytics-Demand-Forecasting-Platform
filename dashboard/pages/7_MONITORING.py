import streamlit as st
import pandas as pd
import numpy as np

# Fix for Evidently/NumPy 2.0 compatibility
if not hasattr(np, "float_"):
    np.float_ = np.float64

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os

st.set_page_config(page_title="Monitoring | RetailPulse", layout="wide")

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

st.title("📡 MLOps Monitoring")
st.caption("Evidently AI-powered data drift, target drift & model performance reports")
st.divider()

if "data" not in st.session_state:
    st.info("⚠️ System Standby: Please upload a dataset in the **Data Upload** portal.")
    st.stop()

df = st.session_state["data"]

try:
    # ── Auto-discover all numeric columns from the actual dataset ──
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("⚠️ Dataset needs at least 2 numeric columns for monitoring analysis.")
        st.stop()

    # First numeric col = target, next up to 4 = features
    target_col   = numeric_cols[0]
    feature_cols = numeric_cols[1:5]

    monitoring_df = df[feature_cols + [target_col]].dropna().copy()

    # Sample for performance if large
    if len(monitoring_df) > 5000:
        monitoring_df = monitoring_df.sample(5000, random_state=42).reset_index(drop=True)

    if len(monitoring_df) < 20:
        st.warning("⚠️ Not enough valid numeric rows after cleaning. Please upload a larger dataset.")
        st.stop()

    monitoring_df = monitoring_df.rename(columns={target_col: 'target'})

    # Split into Reference and Current (Simulated Drift)
    mid = len(monitoring_df) // 2
    reference_data = monitoring_df.iloc[:mid]
    current_data = monitoring_df.iloc[mid:]

    report_type = st.radio("Select Drift Report", ["Data Drift", "Target Drift", "Model Performance"], horizontal=True)

    if st.button("Generate Analytical Report"):
        with st.spinner("Analyzing statistical drift..."):

            if report_type == "Model Performance":
                # ── Manual model performance (avoids sklearn squared= issue) ──
                from sklearn.model_selection import train_test_split
                import tempfile

                X = monitoring_df.drop(columns=['target'])
                y = monitoring_df['target']
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                reg = LinearRegression()
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)

                mae  = mean_absolute_error(y_test, y_pred)
                mse  = np.mean((y_test.values - y_pred) ** 2)
                rmse = np.sqrt(mse)
                r2   = r2_score(y_test, y_pred)

                st.subheader("📊 Model Performance Report")
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE",  f"{mae:.4f}")
                c2.metric("RMSE", f"{rmse:.4f}")
                c3.metric("R²",   f"{r2:.4f}")

                residuals = y_test.values - y_pred
                fig_res = px.scatter(
                    x=y_pred, y=residuals,
                    labels={'x': 'Predicted', 'y': 'Residual'},
                    title="Residual Plot", template="plotly_dark"
                )
                fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_res, use_container_width=True)

                fig_ap = px.scatter(
                    x=y_test.values, y=y_pred,
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    title="Actual vs Predicted", template="plotly_dark"
                )
                fig_ap.add_shape(type='line',
                    x0=float(y_test.min()), y0=float(y_test.min()),
                    x1=float(y_test.max()), y1=float(y_test.max()),
                    line=dict(color='red', dash='dash')
                )
                st.plotly_chart(fig_ap, use_container_width=True)

                # ── Build & offer a self-contained HTML download ──
                html_report = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>RetailPulse – Model Performance Report</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; background:#111; color:#eee; padding:30px; }}
    h1   {{ color:#AB63FA; }} h2 {{ color:#00CC96; margin-top:30px; }}
    .kpis {{ display:flex; gap:40px; margin:20px 0; }}
    .kpi  {{ background:#1e1e2e; border-radius:10px; padding:20px 30px; text-align:center; }}
    .kpi .label {{ font-size:13px; color:#aaa; margin-bottom:6px; }}
    .kpi .value {{ font-size:28px; font-weight:bold; color:#AB63FA; }}
    .chart {{ margin-top:30px; }}
  </style>
</head>
<body>
  <h1>RetailPulse — Model Performance Report</h1>
  <p style="color:#aaa">Linear Regression baseline on {len(monitoring_df):,} samples &nbsp;|&nbsp;
     Target: <code>{target_col}</code> &nbsp;|&nbsp;
     Features: <code>{', '.join(feature_cols)}</code></p>
  <div class="kpis">
    <div class="kpi"><div class="label">MAE</div><div class="value">{mae:.4f}</div></div>
    <div class="kpi"><div class="label">RMSE</div><div class="value">{rmse:.4f}</div></div>
    <div class="kpi"><div class="label">R²</div><div class="value">{r2:.4f}</div></div>
  </div>
  <h2>Residual Plot</h2>
  <div class="chart" id="residual"></div>
  <h2>Actual vs Predicted</h2>
  <div class="chart" id="actpred"></div>
  <script>
    var res_trace = {{
      x: {list(y_pred)},
      y: {list(residuals)},
      mode:'markers', type:'scatter',
      marker:{{color:'#AB63FA', size:5}},
      name:'Residuals'
    }};
    var hline = {{
      x: [{float(y_pred.min())}, {float(y_pred.max())}],
      y: [0, 0], mode:'lines', line:{{color:'red', dash:'dash'}}, name:'Zero'
    }};
    Plotly.newPlot('residual', [res_trace, hline],
      {{title:'Residual Plot', paper_bgcolor:'#111', plot_bgcolor:'#1e1e2e',
        font:{{color:'#eee'}}, xaxis:{{title:'Predicted'}}, yaxis:{{title:'Residual'}}}});

    var ap_trace = {{
      x: {list(y_test.values)},
      y: {list(y_pred)},
      mode:'markers', type:'scatter',
      marker:{{color:'#00CC96', size:5}},
      name:'Predictions'
    }};
    var diag = {{
      x: [{float(y_test.min())}, {float(y_test.max())}],
      y: [{float(y_test.min())}, {float(y_test.max())}],
      mode:'lines', line:{{color:'red', dash:'dash'}}, name:'Perfect fit'
    }};
    Plotly.newPlot('actpred', [ap_trace, diag],
      {{title:'Actual vs Predicted', paper_bgcolor:'#111', plot_bgcolor:'#1e1e2e',
        font:{{color:'#eee'}}, xaxis:{{title:'Actual'}}, yaxis:{{title:'Predicted'}}}});
  </script>
</body>
</html>"""

                st.download_button(
                    label="📥 Download Model Performance HTML Report",
                    data=html_report.encode('utf-8'),
                    file_name="model_performance_drift_report.html",
                    mime="text/html"
                )

            else:
                # ── Evidently Data Drift / Target Drift ───────────
                if report_type == "Data Drift":
                    report = Report(metrics=[DataDriftPreset()])
                else:
                    report = Report(metrics=[TargetDriftPreset()])

                import tempfile
                report_path = os.path.join(tempfile.gettempdir(), "retailpulse_drift_report.html")
                report.run(reference_data=reference_data, current_data=current_data)
                report.save_html(report_path)
                with open(report_path, 'r', encoding='utf-8') as f:
                    html_data = f.read()
                components.html(html_data, height=1000, scrolling=True)

                st.download_button(
                    label="📥 Download HTML Report",
                    data=html_data,
                    file_name=f"retailpulse_{report_type.lower().replace(' ', '_')}.html",
                    mime="text/html"
                )

except Exception as e:
    st.error(f"Monitoring Engine Error: {e}")
    st.info("Tip: Ensure the dataset has numeric columns for statistical drift analysis.")
