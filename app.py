import streamlit as st
import pandas as pd
import joblib

# Load trained model
model_path = "village_model_best.pkl"  # Make sure this file is in the same folder
model = joblib.load(model_path)

st.set_page_config(page_title="VillageConnect Satisfaction Predictor", page_icon="🌾")

st.title("🌾 VillageConnect Satisfaction Score Predictor")
st.write("Fill in the details to predict the **Satisfaction Score** for your village.")

# User Inputs (based on your CSV columns)
region = st.selectbox("Region", ["East", "West", "North", "South"])
service = st.selectbox("Service", ["Electricity", "Internet", "Water", "Healthcare"])
availability = st.number_input("Availability (%)", min_value=0, max_value=100, value=50)
cost = st.number_input("Cost", min_value=0, value=500)
usage_hours = st.number_input("Usage Hours", min_value=0.0, value=5.0, step=0.1)

# Predict Button
if st.button("🔍 Predict Satisfaction Score"):
    input_data = pd.DataFrame({
        "Region": [region],
        "Service": [service],
        "Availability_%": [availability],
        "Cost": [cost],
        "Usage_Hours": [usage_hours]
    })

    prediction = model.predict(input_data)
    st.success(f"Predicted Satisfaction Score: **{prediction[0]:.2f}**")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & XGBoost for the VillageConnect Project")
