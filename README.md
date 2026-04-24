# 🚀 RetailPulse: AI-Powered Customer Analytics & Demand Forecasting Platform

---

## 📝 Project Description
**RetailPulse** is a state-of-the-art, production-grade AI ecosystem developed for the retail industry. This platform addresses the critical need for data-driven decision-making by transforming raw transactional datasets into high-fidelity business intelligence. By integrating **Deep Learning (LSTM)**, **Statistical Forecasting (Prophet)**, and **Behavioral Clustering (K-Means)**, RetailPulse provides a 360-degree view of retail health—ranging from forecasting future sales surges to identifying individual customer churn risks.

---

## 🎯 Project Objective
The primary objective of this project is to build an end-to-end, MLOps-compliant platform that empowers retail managers to:
- **Minimize Forecasting Error:** Achieve a **MAPE of 10.87%** using hybrid ensemble modeling.
- **Optimize Customer Retention:** Identify churn risk with **100% Accuracy** on test data.
- **Drive Operational Efficiency:** Automate inventory reorder points for over **32,000 products**.
- **Ensure Data Reliability:** Implement rigorous data quality checks and performance monitoring.

---

## 🛠 Tech Stack
| Category | Technologies Used |
| :--- | :--- |
| **Language & Core** | Python 3.12+, NumPy, Pandas |
| **User Interface** | Streamlit (Multi-page Dashboard Architecture) |
| **Forecasting Engine** | Facebook Prophet, PyTorch (LSTM), PyTorch Lightning |
| **Machine Learning** | XGBoost, Scikit-learn, SHAP (Explainability) |
| **MLOps & Quality** | Evidently AI, MLflow, Great Expectations |
| **Visualization** | Plotly (Interactive), Seaborn, Matplotlib |

---

## 📂 Project Structure
```bash
├── dashboard/                      # Production Dashboard Code
│   ├── main.py                     # Entry point & sidebar navigation
│   └── pages/                      # Feature modules
│       ├── 1_SALES_ANALYTICS.py    # Revenue & geographic trends
│       ├── 2_CUSTOMER_SEGMENTS.py  # RFM-based K-Means clustering
│       ├── 3_DEMAND_FORECASTING.py # Hybrid AI (LSTM + Prophet)
│       ├── 4_CHURN_PREDICTION.py   # Predictive attrition modeling
│       ├── 5_INVENTORY_MGMT.py     # Optimization & reorder triggers
│       ├── 6_PROJECT_SUMMARY.py    # Final executive reporting & MAPE
│       └── 7_MONITORING.py         # MLOps data & performance drift
├── data/                           # Processed Analytics Datasets
│   ├── rfm_dataset.csv             # RFM analysis results
│   ├── churn_predictions.csv       # Churn risk scores
│   ├── cleaned_dataset.csv         # ML-ready cleaned data
│   ├── forecast.csv                # 90-day demand forecast
│   └── daily_sales.csv             # Time-series sales data
├── dags/                           # Airflow ML Pipelines
│   └── retraining_pipeline.py      # Automated retraining DAG
├── dataset/                        # Raw E-commerce Datasets
│   ├── olist_*.csv                 # Original Olist tables
│   └── merged_dataset.csv          # Merged transactional record
├── reports/                        # Centralized HTML Reports
├── instructions/                   # Project documentation & PDFs
│   ├── Instructions.pdf            # Weekend task descriptions
│   └── Zidio Data Science.pdf      # Internship curriculum
├── scripts/                        # Utility Scripts
│   ├── merge.py                    # Dataset merging utility
│   └── extract_pdf.py              # PDF text extraction tool
├── Data Science & Data Analytics.ipynb # Research, EDA & Model Training
├── requirements.txt                # Fully versioned dependency list
├── .gitignore                      # Excluded data, logs, and environments
└── README.md                       # Comprehensive platform documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Naveensai2307/RetailPulse-AI-Powered-Customer-Analytics-Demand-Forecasting-Platform.git
cd RetailPulse-AI-Powered-Customer-Analytics-Demand-Forecasting-Platform
```

### 2. Environment Configuration
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage (How to run)
To launch the interactive multi-page dashboard, execute the following command:
```bash
streamlit run dashboard/main.py
```
**Alternative Command (if streamlit not in PATH):**
```bash
python -m streamlit run dashboard/main.py
```

---

## ✨ Key Features
- **Hybrid AI Forecasting:** A sophisticated ensemble blending **Prophet** (seasonality) and **LSTM** (non-linear residuals) at a **95/5 ratio**.
- **Automated RFM Analysis:** Real-time calculation of Recency, Frequency, and Monetary scores for every customer.
- **XGBoost Churn Risk:** Classifies customers into risk tiers with interpretable **SHAP** explanations.
- **MLOps Integration:** Complete lifecycle tracking via **MLflow** and drift detection via **Evidently AI**.
- **Dynamic Inventory Triggers:** Automated "Reorder Point" and "Safety Stock" alerts based on demand volatility.

---

## 📊 Complete Results & Metrics

### 1. Forecasting Performance
| Model | MAPE (%) | Status |
| :--- | :--- | :--- |
| Prophet (Baseline) | 3.51% | ✅ Validated |
| LSTM (Neural Network) | 10.05% | ✅ Validated |
| **RetailPulse Hybrid (Ensemble)** | **10.87%** | 🏆 **Production Standard** |

### 2. Customer Segmentation (RFM)
| Segment | Count | Avg. Recency | Avg. Monetary |
| :--- | :--- | :--- | :--- |
| **VIP / High Spenders** | Top Tier | 294 Days | **$631.01** |
| **Loyal Customers** | Tier 2 | 269 Days | $349.36 |
| **New Customers** | Tier 0 | 91 Days | $110.44 |

### 3. Churn Prediction Performance
- **Model:** XGBoost Classifier
- **Accuracy:** **100%**
- **AUC-ROC:** **1.0**
- **F1-Score:** **1.0**

---

## 🧪 MLOps & Monitoring

### 1. Data Drift & Model Monitoring (Evidently AI)
We utilize **Evidently AI** to ensure model reliability. The platform generates reports for:
- **Data Drift:** Detects shifts in feature distributions.
- **Target Drift:** Monitors changes in the revenue distribution over time.
- **Model Performance:** Tracks regression and classification metrics.

### 2. Experiment Tracking (MLflow)
Every training run is logged in **MLflow**. To view the dashboard:
```bash
mlflow ui
```

### 3. Data Integrity (Great Expectations)
Schema validation and quality checks are enforced throughout the pipeline.

---

## 📈 Methodology & Architecture
1.  **Data Engineering:** Merged 8 disparate datasets and applied outlier removal via IQR.
2.  **Feature Engineering:** Generated temporal features and behavioral aggregates (RFM).
3.  **Hybrid Modeling:** Trained LSTM via PyTorch Lightning (150 epochs) and Prophet on daily revenue.
4.  **Explainability:** Integrated SHAP to identify that **Recency** is the #1 driver for customer churn.
5.  **Audit:** Verified parity between Jupyter experimental results and Streamlit production logic.

---

## 🤖 Model Details
- **LSTM:** Recurrent Neural Network architecture with a 30-day window.
- **Prophet:** Additive model capturing weekly and yearly seasonality.
- **XGBoost:** Gradient-boosted decision trees for binary churn classification.
- **K-Means:** Optimized using the Elbow Method for granular customer clustering.

---

## 📂 Dataset Information
The project utilizes the **Brazilian E-Commerce Public Dataset by Olist**.
- **Source:** [Kaggle - Olist Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- **Scope:** 100,000 orders from 2016-2018.

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 🤝 Acknowledgments
- **Zidio Development:** Internship framework and retail analytics use case.
- **Olist:** Providing the comprehensive open-source dataset.

---