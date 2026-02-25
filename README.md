# 🏭 Shift Performance Analytics Platform

## 📌 Project Summary

This project delivers an end-to-end analytics solution for monitoring, analysing, and optimising manufacturing shift performance.

The platform integrates SQL-based data extraction, KPI engineering, machine learning modelling, and an interactive Streamlit dashboard to support operational decision-making.

It enables production managers to:

- Monitor real-time shift KPIs  
- Identify key performance drivers  
- Predict shift efficiency outcomes  
- Optimise workforce and machine configurations  
- Estimate financial impact of efficiency improvements  

---

## 🎯 Business Problem

Manufacturing environments often experience:

- Inconsistent shift productivity  
- High machine downtime  
- Limited visibility into operator performance  
- No predictive insight for future shift outcomes  

This solution transforms raw operational data into actionable intelligence that supports data-driven production management.

---

## 🧱 Data & Processing Pipeline

**Data Source**
- Relational database (SQLite simulation of production records)
- Shift, operator, and machine-level logs

**Processing Steps**
1. SQL-based data extraction  
2. Data cleaning and validation  
3. Feature engineering  
4. KPI computation  
5. Model training and evaluation  
6. Dashboard deployment  

---

## 📊 Key Performance Indicators (KPIs)

The dashboard tracks and visualises:

- Units Produced  
- Defect Count  
- Downtime Percentage  
- Efficiency Score  
- Overall Equipment Effectiveness (OEE)  
- Output per Hour  

---

## 🔍 Analytical Insights

- Identified strong relationship between machine uptime and efficiency score  
- Analysed operator experience vs productivity trends  
- Quantified downtime impact on shift output  
- Segmented high-performing vs underperforming shifts  

---

## 🤖 Machine Learning Model

**Target Variable:**  
Shift Efficiency Score

**Models Tested:**
- Linear Regression  
- Random Forest  
- Gradient Boosting (Selected Model)

**Evaluation Metrics:**
- R²  
- Mean Absolute Error (MAE)

The final model was serialized using joblib and integrated into the Streamlit application for live predictions.

---

## 🔮 Predictive & Optimisation Capabilities

The platform allows users to:

- Predict shift efficiency based on configurable inputs  
- Classify performance category  
- View feature importance  
- Identify optimal shift configurations  
- Estimate ROI from efficiency improvements  

---

## 📊 Streamlit Dashboard Features

### Performance Monitoring
- Real-time KPI summary  
- Shift comparison visualisations  
- Daily performance trend analysis  

### Shift Scoring System
- Weighted performance scoring  
- Component-level breakdown  
- Top-performing shift identification  

### Predictive Analytics
- Efficiency prediction engine  
- Scenario simulation  
- Model performance display  

### Optimisation Engine
- Recommended shift configuration  
- Performance gap analysis  
- Financial impact estimator  

---

## 🧰 Technology Stack

- Python  
- SQL (SQLite)  
- Pandas & NumPy  
- Scikit-learn  
- Plotly  
- Streamlit  
- Joblib  

---

## ⚙️ Setup Instructions

Install dependencies:

pip install streamlit pandas numpy plotly scikit-learn joblib

Run locally:

streamlit run app/streamlit_app.py

Optional:
- If no dataset is detected, synthetic data is generated for demonstration.
- If no trained model is found, predictive features are disabled gracefully.

---

## 📈 Business Value

This project demonstrates how analytics can:

- Improve operational efficiency  
- Support evidence-based scheduling decisions  
- Reduce downtime impact  
- Quantify financial gains from optimisation strategies  

---

## 👤 Author

EFEMINI  
MSc Data Science  
Data Analyst | SQL • Python • KPI Analytics • Machine Learning
