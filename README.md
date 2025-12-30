# 🛡️ NovaPay – AI-Assisted Fraud Detection Platform

An end-to-end **machine learning fraud detection system** for digital money transfers, featuring advanced data engineering, explainable AI, and an interactive Streamlit dashboard for real-time fraud scoring.

---

## 🚀 Project Overview

NovaPay Fraud Detection is a complete data science and AI deployment project that:

- Detects fraudulent digital money transfer transactions  
- Uses **machine learning models** optimized for imbalanced data  
- Provides **explainable predictions** using SHAP  
- Deploys a **modern, AI-assisted web interface** for batch and real-time scoring  

The project follows a **production-ready ML lifecycle**, from raw data ingestion to deployment.

---

## 🎯 Objectives

- Identify high-risk transactions with high precision and recall  
- Understand *why* transactions are flagged as fraud  
- Provide a clean, intuitive interface for non-technical users  
- Demonstrate real-world ML engineering and deployment skills  

---

## 🧠 Solution Architecture

```
Raw Data
   ↓
Data Cleaning & Validation
   ↓
Exploratory Data Analysis (EDA)
   ↓
Feature Engineering
   ↓
Model Training & Evaluation
   ↓
Explainability (SHAP)
   ↓
Deployment (Streamlit AI Dashboard)
```

---

## 📂 Project Structure

```
NovaPay-Fraud-Detection/
│
├── app.py                       # Streamlit AI dashboard
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
├── .streamlit/
│   └── config.toml              # Dark AI-themed UI configuration
│
├── notebooks/
│   ├── Data_Cleaning_NOVAPAY.ipynb
│   ├── EDA_NOVAPAY.ipynb
│   ├── Feature_Engrn_NOVAPAY.ipynb
│   ├── Model_Training_NOVAPAY.ipynb
│   └── SHAP_EXPLANABILITY_RF.ipynb
│
├── models/
│   └── novapay_fraud_model_rf.pkl
│
└── data/
    └── (optional / sample data only)
```

---

## 🧪 Data Processing & Analysis

### 1️⃣ Data Cleaning
- Removed duplicate transaction records  
- Fixed inconsistent categories (channels, currencies, KYC tiers)  
- Handled missing values using **domain-aware logic**  
- Corrected invalid numeric values (negative amounts, scores)  

### 2️⃣ Exploratory Data Analysis (EDA)
Key findings:
- Fraud transactions show **higher transaction velocity**
- Lower **device trust scores** strongly correlate with fraud
- **New accounts** are significantly riskier
- Certain **corridors and channels** carry higher fraud risk
- No data leakage detected

---

## 🧬 Feature Engineering

Engineered features include:
- Transaction velocity ratios
- Account maturity indicators
- Device and IP risk interactions
- Time-based features (hour, day, month)
- Corridor and currency risk indicators

This step significantly improved model performance.

---

## 🤖 Model Training & Evaluation

### Models Tested
- Dummy Classifier (baseline)
- Logistic Regression (balanced)
- Random Forest
- XGBoost

### Final Model
✅ **Random Forest Classifier**

Chosen for:
- Strong ROC-AUC and PR-AUC
- Robust handling of non-linear patterns
- Interpretability with SHAP

---

## 🔍 Explainable AI (SHAP)

To ensure trust and transparency:
- SHAP values were computed for the Random Forest model  
- Global explanations identify **key fraud drivers**
- Local explanations show *why a specific transaction was flagged*

Top drivers:
- Transaction velocity
- Device trust score
- IP risk score
- Account age
- Corridor risk

---

## 🖥️ Deployment – AI-Assisted Dashboard

The system is deployed as a **Streamlit web application** with:

- Batch CSV upload for scoring
- Real-time single transaction scoring
- Adjustable decision threshold
- Model insights and feature importance
- Dark AI-themed dashboard UI
- https://novapay-ai-assisted-fraud-detection-platform-ja3dgckcyfdlb5qlg.streamlit.app/

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📈 Future Improvements

- Real-time API (FastAPI)
- Streaming transaction ingestion
- Auto-retraining pipeline
- Advanced alerting & monitoring

---

## 👤 Author

**Idowu Malachi**  
Data Scientist | Machine Learning Engineer  

---

## ⭐ Final Note

This project demonstrates the full journey from raw data to a deployed, explainable AI system — reflecting real-world fraud detection challenges and solutions.
