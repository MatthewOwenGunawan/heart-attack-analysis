import streamlit as st
import pandas as pd
import joblib

model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/preprocessor.pkl")

st.title("Heart Attack Prediction App")

st.write("Masukkan data pasien di bawah ini:")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0/1)", [0,1])
restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
thalach = st.number_input("Max Heart Rate Achieved")
exang = st.selectbox("Exercise Induced Angina (0/1)", [0,1])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0,1,2,3])
thal = st.selectbox("Thal (0-3)", [0,1,2,3])

if st.button("Predict"):

    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol,
        fbs, restecg, thalach, exang,
        oldpeak, slope, ca, thal
    ]], columns=[
        "age","sex","cp","trestbps","chol",
        "fbs","restecg","thalach","exang",
        "oldpeak","slope","ca","thal"
    ])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Attack (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Heart Attack (Probability: {probability:.2f})")