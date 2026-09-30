import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fraud_model.pkl")
amount_scaler = joblib.load("amount_scaler.pkl")
time_scaler = joblib.load("time_scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Payment Transaction Anomaly Classifier")
st.write("Predicts whether a transaction is legitimate or fraudulent (educational prototype).")

mode = st.sidebar.radio("Input mode", ["Use sample transaction", "Manual entry (Amount & Time)"])

if mode == "Use sample transaction":
    samples = pd.read_csv("sample_transactions.csv")
    idx = st.selectbox("Pick a sample transaction", samples.index)
    row = samples.loc[idx]
    if st.button("Predict"):
        features = row[feature_columns].values.reshape(1, -1)
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]
        st.write(f"Prediction: {'FRAUD' if pred==1 else 'Legitimate'}")
        st.write(f"Fraud probability: {proba:.4f}")
        st.write(f"Actual label: {'FRAUD' if row['Actual']==1 else 'Legitimate'}")
else:
    amount = st.number_input("Transaction Amount", min_value=0.0, value=50.0)
    time_val = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=10000.0)
    if st.button("Predict"):
        row = {col: 0.0 for col in feature_columns}
        row['Amount_scaled'] = amount_scaler.transform([[amount]])[0][0]
        row['Time_scaled'] = time_scaler.transform([[time_val]])[0][0]
        features = pd.DataFrame([row])[feature_columns].values
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]
        st.write(f"Prediction: {'FRAUD' if pred==1 else 'Legitimate'}")
        st.write(f"Fraud probability: {proba:.4f}")
