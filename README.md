🏥 Healthcare Readmission Risk Predictor

End-to-End ML System | FastAPI | MLflow | Docker | Synthetic EHR Data

📌 Overview

This project builds an end-to-end machine learning system that predicts the probability of 30-day hospital readmission using synthetic electronic health record (EHR) data generated via Synthea.

The system includes:

Temporal feature engineering across multiple healthcare tables

Model training and evaluation

Experiment tracking with MLflow

REST API inference service using FastAPI

Dockerized deployment-ready container

This project demonstrates production-oriented ML engineering practices including reproducibility, model artifact management, and containerization.

🎯 Problem Statement

Hospital readmissions are costly and often preventable.
The goal of this system is to predict whether a patient will be readmitted within 30 days of discharge using:

Encounter metadata

365-day medical history

Condition counts

Medication counts

🗂 Project Architecture
healthcare-readmission-ml/
│
├── app/                 # FastAPI inference service
│   ├── main.py
│   ├── predict.py
│   └── schemas.py
│
├── src/                 # Data + training pipeline
│   ├── data_prep.py
│   └── train.py
│
├── models/              # Trained model artifacts
│
├── data/
│   ├── raw/             # Synthea CSV input
│   └── processed/       # Feature-engineered dataset
│
├── mlruns/              # MLflow experiment tracking
│
├── Dockerfile
├── requirements.txt
└── README.md
⚙️ Feature Engineering
Readmission Label Creation

A patient is labeled as readmitted if another encounter occurs within 30 days of discharge.

Engineered Features
Feature	Description
encounter_length_hours	Duration of encounter
encounter_class	Type of encounter
conditions_365d	Number of conditions in prior 365 days
unique_conditions_365d	Unique condition types
meds_365d	Number of medications in prior 365 days
unique_meds_365d	Unique medication types

Temporal joins ensure only historical information is used (no data leakage).

🧠 Model

Logistic Regression (baseline model)

Stratified train/test split

Metrics:

ROC-AUC

Average Precision

Classification report

Experiments are tracked using MLflow.

📊 Experiment Tracking (MLflow)

To launch MLflow UI:

python -m mlflow ui

Open:

http://127.0.0.1:5000

Logged artifacts:

Model parameters

Metrics

Trained model

Classification report

🚀 Running the Project Locally
1️⃣ Create Virtual Environment (Python 3.11 recommended)
py -3.11 -m venv venv
venv\Scripts\activate
2️⃣ Install Dependencies
python -m pip install -r requirements.txt
3️⃣ Prepare Dataset

Place Synthea CSV files inside:

data/raw/
    encounters.csv
    conditions.csv
    medications.csv

Then run:

python src/data_prep.py
4️⃣ Train Model
python src/train.py
🌐 Run Inference API
python -m uvicorn app.main:app --reload

Open:

http://127.0.0.1:8000/docs

Example prediction payload:

{
  "encounter_length_hours": 5.0,
  "encounter_class": "ambulatory"
}
🐳 Docker Deployment
Build Image
docker build -t readmission-api:1.0 .
Run Container
docker run --rm -p 8000:8000 readmission-api:1.0

Then visit:

http://127.0.0.1:8000/docs
🔬 Technical Stack

Python 3.11

Pandas

NumPy

scikit-learn

FastAPI

MLflow

Docker

Uvicorn

🧩 Key Engineering Decisions

Converted all timestamps to UTC-naive to prevent timezone comparison errors

Implemented history-window joins to prevent label leakage

Used MLflow for experiment reproducibility

Containerized inference service for deployment portability

📈 Future Improvements

Upgrade model to XGBoost / LightGBM

Add model drift monitoring

Deploy to Azure App Service

Add CI/CD pipeline for automated retraining

Add feature importance visualization

🏆 What This Project Demonstrates

Temporal feature engineering

Multi-table data integration

Model experiment tracking

Production-style API deployment

Docker containerization

Dependency management and environment control