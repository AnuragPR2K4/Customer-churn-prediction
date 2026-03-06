# Customer-churn-prediction
Machine learning project to predict customer churn using classification models
📊 Telco Customer Churn Prediction
This project focuses on predicting customer churn (whether a customer will leave a telecom service) using machine learning techniques. The aim is to help telecom companies identify customers who are likely to churn and take proactive retention measures.

🚀 Project Overview
Customer churn is a critical business problem in the telecom industry. By analyzing customer demographics, account information, and service usage patterns, this project builds and evaluates multiple machine learning models to accurately predict customer churn.

This project includes:
Data cleaning and preprocessing
Handling class imbalance using SMOTE and Random Undersampling
Training and evaluating multiple machine learning models
Model comparison and selection
Saving the trained model for deployment
A simple Streamlit web application for churn prediction
📁 Dataset
The dataset contains customer-level information including:

Customer Demographics: gender, senior citizen status, dependents
Account Information: tenure, contract type, payment method, billing type
Service Usage: phone service, internet service, streaming services
Target Variable: Churn (Yes / No)
📌 Dataset Source: Telco Customer Churn Dataset

🧩 Features Used
The key features used for model training are:

customerID
gender
SeniorCitizen
Partner
Dependents
tenure
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
MonthlyCharges
TotalCharges
Churn
⚙️ Data Preprocessing
The following preprocessing steps were applied:

Handling missing values
Encoding categorical variables
Feature scaling where required
Handling imbalanced data using:
SMOTE (Synthetic Minority Oversampling Technique)
Random Undersampling of the majority class
📌 Note: Minor FutureWarnings from scikit-learn may appear due to library updates; these do not affect model performance.

🤖 Machine Learning Models Used
The following machine learning models were trained and evaluated:

Logistic Regression
SVM
Random Forest
AdaBoost
Gradient Boosting
XGBoost
Each model was trained on the processed dataset and evaluated using standard classification metrics.

📈 Evaluation Metrics
Models were evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
The best-performing model was selected and saved as a .pkl file for deployment.

🧪 Model Training Notebook
All steps including data analysis, preprocessing, imbalance handling, model training, and evaluation are documented in:

📘 main.ipynb

🌐 Web Application
A simple Streamlit web application is included to predict customer churn based on user input.

📄 File: app.py

To run the app locally:
streamlit run app.py



