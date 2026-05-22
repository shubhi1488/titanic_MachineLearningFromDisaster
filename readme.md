# Titanic Survival Prediction using Machine Learning

A complete end-to-end Machine Learning project that predicts passenger survival on the Titanic dataset using feature engineering, Random Forest, XGBoost, and Gradio deployment.

---

## Project Overview

This project analyzes Titanic passenger data and predicts whether a passenger survived or not based on:
- Passenger Class
- Gender
- Age
- Fare
- Family Information
- Embarkation Port
- Passenger Title

The project includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning model training
- Hyperparameter tuning
- Gradio web application deployment

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Gradio

---

## Machine Learning Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

---

## Dataset

Dataset used:
- Titanic Dataset from Kaggle

Competition Link:
https://www.kaggle.com/competitions/titanic

---

## Project Workflow

```text
Data Loading
→ Data Cleaning
→ Exploratory Data Analysis
→ Feature Engineering
→ Encoding
→ Model Training
→ Evaluation
→ Hyperparameter Tuning
→ Prediction
→ Gradio Deployment
```

---

## Feature Engineering

Created additional features:
- FamilySize
- IsAlone
- Title Extraction

---

## Best Model

Final model used:
- Random Forest Classifier

Achieved accuracy:
- ~80% to 86%

---

## Gradio Web Application

The project includes a Gradio-based web interface where users can:
- Enter passenger details
- Predict survival probability

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run Gradio App

```bash
python app/gradio_app.py
```

---

## Project Structure

```text
Titanic-Survival-Prediction/
│
├── data/
├── notebooks/
├── models/
├── app/
├── outputs/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Future Improvements

- SHAP Explainability
- Streamlit Deployment
- Ensemble Learning
- Docker Deployment
- CI/CD Pipeline

---

## Author

Shubhi Agarwal
