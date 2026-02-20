import streamlit as st
import pandas as pd
import joblib

# load model and scaler
model = joblib.load("model/heart_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient details to predict heart disease")

# INPUT FIELDS
age = st.number_input("Age", 1, 120, 30)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0,1])
chest_pain_type = st.number_input("Chest Pain Type (0-3)",0,3,1)
resting_blood_pressure = st.number_input("Resting Blood Pressure",80,200,120)
cholesterol = st.number_input("Cholesterol",100,400,200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar >120 (1=True,0=False)",[0,1])
resting_ecg = st.number_input("Rest ECG (0-2)",0,2,1)
max_heart_rate = st.number_input("Max Heart Rate",60,220,150)
exercise_induced_angina = st.selectbox("Exercise Induced Angina (1=yes,0=no)",[0,1])
st_depression = st.number_input("ST Depression",0.0,6.0,1.0)
st_slope = st.number_input("ST Slope (0-2)",0,2,1)
num_major_vessels = st.number_input("Major Vessels (0-4)",0,4,0)
thalassemia = st.number_input("Thalassemia (0-3)",0,3,1)

# PREDICTION BUTTON
if st.button("Predict"):
    
    input_data = pd.DataFrame([[age,sex,chest_pain_type,resting_blood_pressure,
                                cholesterol,fasting_blood_sugar,resting_ecg,max_heart_rate,
                                exercise_induced_angina,st_depression,st_slope,
                                num_major_vessels,thalassemia]],
                              
    columns=['age','sex','chest_pain_type','resting_blood_pressure','cholesterol',
             'fasting_blood_sugar','resting_ecg','max_heart_rate',
             'exercise_induced_angina','st_depression','st_slope',
             'num_major_vessels','thalassemia'])

    # scale input
    input_scaled = scaler.transform(input_data)

    # predict
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High chance of Heart Disease")
    else:
        st.success("✅ Low chance of Heart Disease")
