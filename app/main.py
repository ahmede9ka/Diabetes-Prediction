import streamlit as st
import numpy as np
import pickle

# Load the trained SVM model
with open('../diabetes.pkl', 'rb') as file:
    model = pickle.load(file)

# Title of the app
st.title("Diabetes Prediction App")

# Input fields for user data
st.header("Enter the patient's information:")
pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0.0, step=1.0)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, step=1.0)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, step=1.0)
insulin = st.number_input("Insulin Level (μU/mL)", min_value=0.0, step=1.0)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
age = st.number_input("Age", min_value=1, step=1)

# Collect input data
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    print("hello")
    if prediction[0] == 1:
        st.error("The patient is likely to have diabetes.")
    else:
        st.success("The patient is not likely to have diabetes.")
