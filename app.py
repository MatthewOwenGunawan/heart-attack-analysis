import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("artifacts/preprocessor.pkl")
model = joblib.load("artifacts/model.pkl")

FEATURES = [
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak",
    "slope","ca","thal"
]

def make_prediction(input_data):

    input_array = np.array(input_data).reshape(1, -1)

    scaled_data = scaler.transform(input_array)

    pred = model.predict(scaled_data)[0]

    return pred


def main():

    st.title("Heart Attack Prediction App")

    st.write("Masukkan nilai fitur di bawah")

    inputs = []

    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    inputs.append(age)

    sex = st.selectbox("Sex (0 = female, 1 = male)", [0,1])
    inputs.append(sex)

    cp = st.number_input("Chest Pain Type (cp)", 0,3,1)
    inputs.append(cp)

    trestbps = st.number_input("Resting Blood Pressure", 0,300,120)
    inputs.append(trestbps)

    chol = st.number_input("Cholesterol", 0,600,200)
    inputs.append(chol)

    fbs = st.selectbox("Fasting Blood Sugar > 120?", [0,1])
    inputs.append(fbs)

    restecg = st.number_input("Rest ECG", 0,2,1)
    inputs.append(restecg)

    thalach = st.number_input("Max Heart Rate", 0,250,150)
    inputs.append(thalach)

    exang = st.selectbox("Exercise Induced Angina", [0,1])
    inputs.append(exang)

    oldpeak = st.number_input("Oldpeak", 0.0,10.0,1.0)
    inputs.append(oldpeak)

    slope = st.number_input("Slope", 0,2,1)
    inputs.append(slope)

    ca = st.number_input("Number of major vessels (ca)", 0,4,0)
    inputs.append(ca)

    thal = st.number_input("Thal", 0,3,2)
    inputs.append(thal)

    if st.button("Predict Heart Attack Risk"):

        result = make_prediction(inputs)

        if result == 1:
            st.error("High Risk of Heart Attack")
        else:
            st.success("Low Risk of Heart Attack")


if __name__ == "__main__":
    main()