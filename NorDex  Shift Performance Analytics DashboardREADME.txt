🏭 Project Overview
This Streamlit dashboard provides real-time monitoring, predictive analytics, and optimization tools for shift performance in a manufacturing environment. It enables production managers and analysts to:

Track key performance indicators (KPIs)

Analyze operator and machine efficiency

Predict shift outcomes using machine learning

Identify optimal configurations for improved productivity

🚀 Features
📊 Performance Dashboard
Real-time KPIs: Units produced, defect count, downtime, efficiency score, and OEE

Shift comparison charts and experience-performance analysis

Daily performance trend visualization

🎯 Shift Scoring Model
Weighted scoring system based on production, quality, and availability

Score distribution and component breakdown

Top-performing shift identification

🔮 Predictive Analytics
Predict efficiency score based on user-defined shift parameters

Performance category classification

Feature importance and model accuracy visualization

⚙️ Optimization Engine
Optimal shift configuration finder using ML model

Actionable recommendations based on performance gaps

ROI calculator for estimating financial impact of efficiency improvements

🧰 Technology Stack
Frontend: Streamlit

Data Processing: Pandas, NumPy

Visualization: Plotly (Express & Graph Objects)

Model Deployment: joblib (pickle serialization)

Backend: Python

Database: CSV (or SQL Server if integrated)

📂 File Structure
Code
├── app.py                      # Main Streamlit application
├── v_shiftPerformance1.csv     # Shift performance dataset (optional)
├── best_model.pkl              # Trained ML model (optional)
├── README.pdf                  # Project documentation
⚙️ Setup Instructions
Install dependencies:

bash
pip install streamlit pandas numpy plotly scikit-learn joblib
Run the app:

bash
streamlit run app.py
Optional files:

Place v_shiftPerformance1.csv in the root directory for real data

Place best_model.pkl for predictive features

📈 Model Details
Algorithm: Gradient Boosting (100 estimators)

Preprocessing: OneHotEncoding for categorical variables

Validation: Train/test split with cross-validation

Deployment: Integrated into Streamlit for live predictions

📌 Notes
If no CSV is found, the app generates synthetic data for demonstration

If no model is found, predictive features are disabled with a warning

All filters and inputs are interactive via the sidebar and tabs

👤 Author
EFEMINI
Data Scientist | Analytics & Optimization Specialist