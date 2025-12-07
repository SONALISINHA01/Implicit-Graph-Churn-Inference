import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("best_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📊 Telco Customer Churn Prediction App")
st.write("Enter customer details below to predict churn probability.")

def user_input_features():

    # Numerical Inputs
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    tenure = st.number_input("Tenure (Months)", 0, 100, 12)
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.35)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

    # Gender
    Female = st.selectbox("Gender (Female=1, Male=0)", [0, 1])

    # Simple yes/no inputs
    Partner_Yes = st.selectbox("Partner Yes", [0, 1])
    Dependents_Yes = st.selectbox("Dependents Yes", [0, 1])
    PhoneService_Yes = st.selectbox("Phone Service Yes", [0, 1])

    # MULTIPLE LINES
    ml_option = st.selectbox("Multiple Lines", ["No phone service", "Yes", "No"])
    MultipleLines_No_phone_service = 1 if ml_option == "No phone service" else 0
    MultipleLines_Yes = 1 if ml_option == "Yes" else 0

    # INTERNET SERVICE
    internet_option = st.selectbox("Internet Service", ["Fiber optic", "DSL/Other", "No"])
    InternetService_Fiber_optic = 1 if internet_option == "Fiber optic" else 0
    InternetService_No = 1 if internet_option == "No" else 0

    # ONLINE SECURITY, BACKUP, DEVICE PROTECTION, TECH SUPPORT, STREAMING
    # Each one has three possible values:
    # "Yes" / "No" / "No internet service"

    def three_option_input(label):
        opt = st.selectbox(label, ["Yes", "No", "No internet service"])
        return (
            1 if opt == "No internet service" else 0,  # _No internet service
            1 if opt == "Yes" else 0                   # _Yes
        )

    OnlineSecurity_No_int, OnlineSecurity_Yes = three_option_input("Online Security")
    OnlineBackup_No_int, OnlineBackup_Yes = three_option_input("Online Backup")
    DeviceProtection_No_int, DeviceProtection_Yes = three_option_input("Device Protection")
    TechSupport_No_int, TechSupport_Yes = three_option_input("Tech Support")
    StreamingTV_No_int, StreamingTV_Yes = three_option_input("Streaming TV")
    StreamingMovies_No_int, StreamingMovies_Yes = three_option_input("Streaming Movies")

    # CONTRACT
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    Contract_One_year = 1 if contract == "One year" else 0
    Contract_Two_year = 1 if contract == "Two year" else 0

    # Paperless Billing
    PaperlessBilling_Yes = st.selectbox("Paperless Billing Yes", [0, 1])

    # Payment Method
    payment = st.selectbox("Payment Method", [
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ])

    Payment_Credit_auto = 1 if payment == "Credit card (automatic)" else 0
    Payment_Electronic_check = 1 if payment == "Electronic check" else 0
    Payment_Mailed_check = 1 if payment == "Mailed check" else 0

    # BUILD THE FINAL DICTIONARY EXACTLY MATCHING TRAINING COLUMNS
    data = {
        'SeniorCitizen': SeniorCitizen,
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'Churn_Yes': 0,   # always 0 for new input

        'Female': Female,
        'Partner_Yes': Partner_Yes,
        'Dependents_Yes': Dependents_Yes,
        'PhoneService_Yes': PhoneService_Yes,

        'MultipleLines_No phone service': MultipleLines_No_phone_service,
        'MultipleLines_Yes': MultipleLines_Yes,

        'InternetService_Fiber optic': InternetService_Fiber_optic,
        'InternetService_No': InternetService_No,

        'OnlineSecurity_No internet service': OnlineSecurity_No_int,
        'OnlineSecurity_Yes': OnlineSecurity_Yes,

        'OnlineBackup_No internet service': OnlineBackup_No_int,
        'OnlineBackup_Yes': OnlineBackup_Yes,

        'DeviceProtection_No internet service': DeviceProtection_No_int,
        'DeviceProtection_Yes': DeviceProtection_Yes,

        'TechSupport_No internet service': TechSupport_No_int,
        'TechSupport_Yes': TechSupport_Yes,

        'StreamingTV_No internet service': StreamingTV_No_int,
        'StreamingTV_Yes': StreamingTV_Yes,

        'StreamingMovies_No internet service': StreamingMovies_No_int,
        'StreamingMovies_Yes': StreamingMovies_Yes,

        'Contract_One year': Contract_One_year,
        'Contract_Two year': Contract_Two_year,

        'PaperlessBilling_Yes': PaperlessBilling_Yes,

        'PaymentMethod_Credit card (automatic)': Payment_Credit_auto,
        'PaymentMethod_Electronic check': Payment_Electronic_check,
        'PaymentMethod_Mailed check': Payment_Mailed_check
    }

    return pd.DataFrame([data])

input_df = user_input_features()

# Scale input
scaled_input = scaler.transform(input_df)

# Predict
prediction = model.predict(scaled_input)[0]
prob = model.predict_proba(scaled_input)[0][1]

st.subheader("🔍 Prediction:")
if prediction == 1:
    st.error("❌ The customer is likely to CHURN")
else:
    st.success("✅ The customer is NOT likely to churn")

st.subheader("📈 Churn Probability:")
st.write(f"**{prob:.2f}**")
