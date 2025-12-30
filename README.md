# 🛡️ NovaPay Fraud Detection System

An end-to-end **AI-driven fraud detection system** for digital money transfers, designed to identify, explain, and manage fraudulent transactions with high precision and strong recall.

This project demonstrates the **complete data science lifecycle** — from raw data cleaning to model explainability and deployment — using real-world fraud detection logic.

---

## 📌 Project Overview

Digital payment platforms face increasing fraud risks due to:
- High transaction volumes
- Cross-border transfers
- Device and IP spoofing
- Rapid transaction velocity

The **NovaPay Fraud Detection System** was built to address these challenges by:

- Cleaning and standardizing messy transactional data  
- Engineering fraud-relevant behavioral features  
- Training and comparing multiple machine learning models  
- Prioritizing **precision-critical fraud detection**  
- Explaining model decisions using SHAP  
- Providing an interactive Streamlit interface for analysts  

---

## 🎯 Objectives

- Detect fraudulent transactions with **very high precision**
- Minimize false positives (critical in financial systems)
- Provide **interpretable fraud decisions**
- Allow analysts to review, score, and monitor transactions
- Demonstrate production-ready ML practices

---

## 🧠 Machine Learning Approach

### Models Trained & Evaluated
The project trains and compares **8 classification models**, including:

- Random Forest ⭐ (best performing)
- XGBoost
- Logistic Regression
- Decision Tree
- Gradient Boosting
- AdaBoost
- K-Nearest Neighbors
- Support Vector Machine

Each model is evaluated using:
- Precision
- Recall
- F1-Score
- ROC-AUC

> **Key Result:**  
> The **Random Forest model** achieved **precision ≈ 1.00** while maintaining strong recall, making it the preferred model for deployment.

---

## 📂 Project Structure

NovaPay-Fraud-Detection-AI-Platform/
│
├── Data/
│ ├── nova_pay_transcations.csv
│ ├── nova_pay_fraud_boost.csv
│ └── Nova_cleaned_df.csv
│
├── notebooks/
│ ├── Data_Cleaning.ipynb
│ ├── EDA.ipynb
│ ├── Feature_Engineering.ipynb
│ ├── Model_Training_Comparison.ipynb
│ └── SHAP_Explainability.ipynb
│
├── models/
│ ├── rf_fraud_model.joblib
│ ├── rf_features.json
│ └── shap_values_rf.npy
│
├── streamlit.py
├── requirements.txt
└── README.md

---

## 🧹 Data Cleaning Highlights

- Combined multiple transaction datasets
- Removed duplicates and invalid records
- Standardized inconsistent categories:
  - Channels (ATM / WEB / MOBILE)
  - Countries
  - KYC tiers
- Corrected invalid ranges:
  - Negative transaction amounts
  - Out-of-range risk scores
- Intelligently handled missing values
- Converted all data types correctly

Result: **A clean, modeling-ready dataset with 11,000+ transactions**

---

## 📊 Exploratory Data Analysis (EDA)

Key fraud patterns discovered:

- Fraudulent transactions show **higher IP risk scores**
- **Low device trust scores** strongly correlate with fraud
- Fraudsters perform **many transactions in short time windows**
- Newer accounts are more likely to be fraudulent
- Certain corridors and channels carry elevated risk

These insights directly informed feature engineering and model design.

---

## ⚙️ Feature Engineering

Advanced features include:

- Transaction velocity (1h, 24h)
- Account age behavior
- Device and IP trust indicators
- Corridor risk signals
- Time-based features (hour, day, weekend)
- Aggregated behavioral metrics

Feature engineering was designed to **mimic real fraud analyst logic**, not just statistical transformations.

---

## 🔍 Model Explainability (SHAP)

To ensure transparency and trust:

- Global SHAP feature importance
- Beeswarm plots for feature impact
- Dependence plots for key drivers
- Force plots for individual fraud decisions

This allows:
- Analyst validation
- Regulatory defensibility
- Clear explanation of why a transaction was flagged

---

## 🖥️ Deployment (Streamlit App)

The Streamlit application provides:

- Batch transaction scoring
- Fraud probability and risk tiers
- Precision-focused fraud flagging
- Analyst-friendly dashboards
- Explainability integration (optional)

Designed to simulate a **real financial fraud operations console**.

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
pip install -r requirements.txt
2️⃣ Run the Streamlit App
streamlit run streamlit.py
3️⃣ Open in Browser
http://localhost:8501

📈 Key Takeaways
Fraud detection requires precision-first thinking

Model explainability is essential in financial systems

Feature engineering matters more than model complexity

Random Forest provided the best balance of accuracy and stability

The project is fully extensible to real-world production systems

📬 Contact
Author: Idowu Malachi
📧 Email: idowumalachi@gmail.com

For questions, feedback, or collaboration opportunities, feel free to reach out.

🏁 Final Note
This project was developed to reflect industry-grade fraud detection standards, combining strong analytics, sound ML practices, and real deployment considerations.
