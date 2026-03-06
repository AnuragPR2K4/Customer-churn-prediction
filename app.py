# Imports
import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# Load trained objects
# -----------------------------
model = pickle.load(open("gradient_boosting_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📉 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn")

# -----------------------------
# User Inputs
# -----------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner?", ["Yes", "No"])
Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"]
)

MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=2000.0)

# -----------------------------
# Encoding (same logic as training)
# -----------------------------
def bin_encode(val):
    return 1 if val == "Yes" else 0

data = {
    "gender": 1 if gender == "Male" else 0,
    "SeniorCitizen": SeniorCitizen,
    "Partner": bin_encode(Partner),
    "Dependents": bin_encode(Dependents),
    "tenure": tenure,
    "PhoneService": bin_encode(PhoneService),
    "PaperlessBilling": bin_encode(PaperlessBilling),
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
}

# Create DataFrame
input_df = pd.DataFrame([data])

# -----------------------------
# 🔥 THIS IS THE FIX 🔥
# -----------------------------
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Scale
input_scaled = scaler.transform(input_df)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Churn"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"⚠️ Customer is likely to CHURN (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer is NOT likely to churn (Probability: {1 - prob:.2f})")
