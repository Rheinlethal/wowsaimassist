import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("offset_predictor.pkl")

st.title("Ini Aim Assist")

# Input form
shell_travel_time = st.number_input("Shell travel time (s)", value=0)
distance = st.number_input("Distance (km)", value=0)
angle = st.number_input("Angle (deg)", value=0)
enemy_max_speed = st.number_input("Enemy max speed (knots)", value=0)

# Predict button
if st.button("Predict Offset_X"):
    X = np.array([[shell_travel_time, distance, angle, enemy_max_speed]])
    pred = model.predict(X)[0]
    st.success(f"Predicted offset_x: {pred:.2f}")