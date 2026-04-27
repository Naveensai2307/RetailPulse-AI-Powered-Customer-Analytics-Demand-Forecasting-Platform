# 🚀 RetailPulse: AI-Powered Customer Analytics & Demand Forecasting Platform

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://retailpulse-ai-powered-customer-analytics-demand-forecasting-p.streamlit.app/)

---

## 📝 Project Description
**RetailPulse** is a state-of-the-art, production-grade AI ecosystem developed for the retail industry. This platform addresses the critical need for data-driven decision-making by transforming raw transactional datasets into high-fidelity business intelligence. By integrating **Deep Learning (LSTM)**, **Statistical Forecasting (Prophet)**, and **Behavioral Clustering (K-Means)**, RetailPulse provides a 360-degree view of retail health—ranging from forecasting future sales surges to identifying individual customer churn risks.

---

## 🛠 Tech Stack
| Category | Technologies Used |
| :--- | :--- |
| **Language & Core** | Python 3.12+, NumPy, Pandas |
| **User Interface** | Streamlit (Multi-page Dashboard Architecture) |
| **Forecasting Engine** | Prophet, PyTorch (LSTM), PyTorch Lightning |
| **Machine Learning** | XGBoost, Scikit-learn, SHAP (Explainability) |
| **MLOps & Quality** | Evidently AI, MLflow, Great Expectations |
| **Visualization** | Plotly (Interactive), Seaborn, Matplotlib |
| **DevOps & CI/CD** | Docker (Multi-stage), Kubernetes, GitHub Actions |
| **Observability** | Prometheus, Grafana |

---

## 📂 Project Structure
```bash
├── .github/                       # CI/CD Workflows (GitHub Actions)
├── .streamlit/                    # Streamlit Theme & Server Config
├── dashboard/                      # Production Dashboard Code
│   ├── main.py                     # Entry point & metrics exporter
│   └── pages/                      # Feature modules (Sales, Forecasting, Monitoring)
├── monitoring/                    # Prometheus & Grafana Configs
│   ├── prometheus.yml              # Scrape configuration
│   ├── grafana-dashboard.json      # Pre-configured visualization
│   └── docker-compose.monitoring.yml # Monitoring stack orchestration
├── kubernetes/                    # K8s Orchestration Manifests
│   ├── deployment.yaml             # 3-replica production deployment
│   └── service.yaml                # LoadBalancer configuration
├── dags/                           # Airflow ML Pipelines
│   └── retraining_pipeline.py      # Automated retraining DAG
├── data/                           # Processed Analytics Datasets
├── dataset/                        # Raw E-commerce Datasets (Olist)
├── instructions/                   # Zidio Internship Guidelines & PDFs
├── mlruns/                         # MLflow Experiment Tracking Database
├── reports/                        # Centralized HTML Analysis Reports
├── scripts/                        # Utility & Validation Scripts
│   ├── load_test.py                # Concurrent user performance script
│   └── merge.py                    # Dataset merging utility
├── requirements.txt                # Fully versioned dependency list
├── Dockerfile                      # Multi-stage production build
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
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Usage (Run Commands)

### 📊 Interactive Dashboard
Launch the multi-page platform locally:
```bash
python -m streamlit run dashboard/main.py
```

### ☁️ Live Deployed Version
Access the platform instantly on **Streamlit Community Cloud**:
👉 **[RetailPulse Live Dashboard](https://retailpulse-ai-powered-customer-analytics-demand-forecasting-p.streamlit.app/)**

### 🧪 Performance Validation
Run the concurrency load test to verify sub-second latency:
```bash
python scripts/load_test.py
```

---

## 🐳 Production Deployment (DevOps)

### 1. Docker Production Run
Build the optimized multi-stage image:
```bash
docker build -t retailpulse:latest .
docker run -p 8501:8501 retailpulse:latest
```

### 2. Kubernetes Orchestration
Deploy to a cluster with high-availability (3 replicas):
```bash
kubectl apply -f kubernetes/
```

### 3. Automated CI/CD
Every push to `main` triggers a GitHub Action to build and push the production image to **GHCR (GitHub Container Registry)**.

---

## 📡 Monitoring & Observability (MLOps)
RetailPulse features a dedicated observability stack for 24/7 health tracking.

### Launch Prometheus & Grafana
```bash
docker-compose -f monitoring/docker-compose.monitoring.yml up -d
```
- **Prometheus:** [http://localhost:9090](http://localhost:9090) (Scrapes metrics from port 8001).
- **Grafana:** [http://localhost:3000](http://localhost:3000) (Import `monitoring/grafana-dashboard.json`).

---

## 🔄 MLOps Operations

### 1. Experiment Tracking (MLflow)
View all model training logs and performance metrics:
```bash
mlflow ui
```
*Access at: `http://localhost:5000`*

### 2. Automated Retraining (Apache Airflow)
Set up the retraining pipeline (WSL/Ubuntu recommended):
```bash
export AIRFLOW_HOME=$(pwd)
airflow db init
airflow scheduler & airflow webserver --port 8080
```
*Trigger the `zidio_model_retraining_pipeline` at `http://localhost:8080`*

---

## 📊 Performance & Accuracy Metrics

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

### 3. Predictive Performance
- **Churn Model:** XGBoost Classifier
- **Churn Accuracy:** **100%** | **AUC-ROC:** **1.0**
- **Load Test Success:** **100%** | **Avg Latency:** **744ms**

---

## ✨ Key Technical Features
- **Hybrid AI Forecasting:** Ensemble blending **Prophet** (seasonality) and **LSTM** (neural residuals).
- **Automated RFM Analysis:** Real-time customer behavioral clustering.
- **Explainable AI (XAI):** Integrated **SHAP** values for churn driver transparency.
- **MLOps Drift Detection:** **Evidently AI** integration for statistical data validation.
- **Dynamic Inventory Triggers:** Automated reorder point calculation based on forecast volatility.

---

## 📂 Dataset Information
The project utilizes the **Brazilian E-Commerce Public Dataset by Olist**.
- **Scope:** 100,000 orders from 2016-2018.
- **Source:** [Kaggle - Olist Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

---

## 🤝 Acknowledgments
- **Zidio Development:** Internship framework and retail analytics use case.
- **Olist:** Providing the comprehensive transactional dataset.

---

