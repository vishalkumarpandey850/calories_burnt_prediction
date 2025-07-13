import streamlit  as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# App title
st.title("Calories Burnt Prediction App")
st.markdown("Estimate how many calories you burn based on exercise and personal info.")

# Sidebar for input
st.sidebar.header("📝 Input Features")
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
age = st.sidebar.slider("Age", 10, 80, 25)
height = st.sidebar.slider("Height (cm)", 120, 210, 170)
weight = st.sidebar.slider("Weight (kg)", 30, 150, 70)
duration = st.sidebar.slider("Duration (min)", 5, 120, 30)
heart_rate = st.sidebar.slider("Heart Rate", 60, 180, 100)
body_temp = st.sidebar.slider("Body Temperature (°C)", 35.0, 42.0, 37.0)

# Convert gender to numeric
gender_val = 0 if gender == "Female" else 1

# Prediction
input_data = np.array([[gender_val, age, height, weight, duration, heart_rate, body_temp]])
prediction = model.predict(input_data)[0]

st.subheader("✅ Prediction Result")
st.success(f"Estimated Calories Burnt: **{prediction:.2f}** kcal")

# Optional: Show model details
if st.checkbox("Show Model Inputs"):
    st.write(pd.DataFrame(input_data, columns=[
        "Gender", "Age", "Height", "Weight", "Duration", "Heart Rate", "Body Temp"
    ]))
