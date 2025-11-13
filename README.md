🩺 Patient Health API – CRUD & Diabetes Prediction
A web-based system to manage patient records and predict diabetes risk using a trained Machine Learning model. This project provides a FastAPI backend for CRUD operations and a Streamlit frontend for an interactive dashboard.

📌 Project Overview
Project Name: Patient Health API – CRUD Testing with ML Prediction

Purpose:

Store and manage patient records (Create, Read, Update, Delete)
Predict diabetes risk using patient health metrics
Display dashboards, charts, and health summaries
Features:

Dashboard with total patients, prediction stats, and recent patients
Add new patients with a user-friendly form
Edit and delete existing patients
Run ML-based diabetes predictions
Health check endpoint to verify system status
⚙️ Tools and Technologies
Python – programming language
FastAPI – backend API
Streamlit – frontend web app
Pandas & Plotly – data handling and visualization
Joblib & NumPy – load ML model and scaler
JSON file (patients.json) – local database for patient records
ML Model:

Gradient Boosting Classifier
Features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
Outputs: Positive or Negative with probability score
🚀 Setup and Installation
Clone the repository
git clone https://github.com/JamshedAli18/Patient-Health-API-CRUD-Testing-with-ML-Prediction.git
cd <your-repo-folder>

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


pip install -r requirements.txt
# If requirements.txt not provided:
pip install fastapi uvicorn streamlit pandas plotly scikit-learn joblib imbalanced-learn


uvicorn main:app --reload


streamlit run streamlit_app.py
