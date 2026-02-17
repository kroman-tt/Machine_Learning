import streamlit as st
import numpy as np
import joblib
import pandas as pd

model = joblib.load("heart_disease_classfier.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")


st.title("Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease risk.")

# Numeric input
age = st.number_input("Age", min_value=1, max_value=120, value=50)
resting_bp = st.number_input(
    "Resting_Blood_Pressure", min_value=50, max_value=250, value=120
)
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
max_hr = st.number_input("Max_Heart_Rate", min_value=50, max_value=220, value=150)
st_depression = st.number_input(
    "ST_Depression", min_value=0.0, max_value=10.0, value=1.0
)

# Categorical inputs
gender = st.selectbox("Gender", ["Male", "Female"])
chest_pain = st.selectbox(
    "Chest_Pain_Type",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
)
fasting_bs = st.selectbox("Fasting_Blood_Sugar", ["Normal", "High"])
resting_ecg = st.selectbox(
    "Resting_ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
)
exercise_angina = st.selectbox("Exercise_Induced_Angina", ["Yes", "No"])
st_slope = st.selectbox("ST_Slope", ["Upsloping", "Flat", "Downsloping"])
vessels = st.selectbox("Number_of_Vessels", ["Zero", "One", "Two", "Three", "Four"])
thal = st.selectbox(
    "Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect", "Not Described"]
)

### prediction
if st.button("Predict"):
    # input_data = # Create input dataframe
    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Resting_Blood_Pressure": [resting_bp],
            "Cholesterol": [cholesterol],
            "Max_Heart_Rate": [max_hr],
            "ST_Depression": [st_depression],
            "Gender": [gender],
            "Chest_Pain_Type": [chest_pain],
            "Fasting_Blood_Sugar": [fasting_bs],
            "Resting_ECG": [resting_ecg],
            "Exercise_Induced_Angina": [exercise_angina],
            "ST_Slope": [st_slope],
            "Number_of_Vessels": [vessels],
            "Thalassemia": [thal],
        }
    )

    input_encoded = pd.get_dummies(
        input_data,
        columns=[
            "Gender",
            "Chest_Pain_Type",
            "Fasting_Blood_Sugar",
            "Resting_ECG",
            "Exercise_Induced_Angina",
            "ST_Slope",
            "Number_of_Vessels",
            "Thalassemia",
        ],
        drop_first=False,
        dtype=int,
    )
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Show result
    if prediction == 1:
        st.error(f"High Risk of Heart Disease (Probability: {probability:.2f})")
    else:
        st.success(f"Low Risk of Heart Disease (Probability: {probability:.2f})")
