#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import joblib

# Load saved model and scaler
model = joblib.load("xgb_injury_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Injury Risk Prediction", layout="centered")
st.title("🏥 Injury Risk Prediction for Athletes")

# Input form
age = st.number_input("Age", min_value=16, max_value=45, value=25)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=70.0)
height = st.number_input("Height (cm)", min_value=140.0, max_value=220.0, value=175.0)
previous_injuries = st.selectbox("Previous Injuries", [0, 1])
training_intensity = st.slider("Training Intensity", 0.0, 1.0, 0.5)
recovery_time = st.slider("Recovery Time (days)", 0, 10, 3)

if st.button("Predict Injury Risk"):
    input_data = pd.DataFrame([[age, weight, height, previous_injuries, training_intensity, recovery_time]],
                              columns=["Player_Age", "Player_Weight", "Player_Height",
                                       "Previous_Injuries", "Training_Intensity", "Recovery_Time"])
    
    # Scale features
    input_scaled = scaler.transform(input_data)

    # Predict
    pred = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)[0][1]

    if pred[0] == 1:
        st.error(f"⚠️ High Injury Risk! Probability: {proba:.2f}")
    else:
        st.success(f"✅ Low Injury Risk. Probability: {proba:.2f}")


# In[ ]:





# In[ ]:




