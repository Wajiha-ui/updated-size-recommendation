import streamlit as st
import joblib
import numpy as np
import os

# Load the trained model
def load_model():
    model_path = os.path.join(os.getcwd(), "best_size_model.pkl")  # Ensure correct filename
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

# Initialize model
model = load_model()

# Streamlit UI
st.title("AI Size Recommendation Model")
st.write("Enter your details to get the recommended clothing size.")

# User Inputs
height = st.number_input("Enter your height (cm):", min_value=100, max_value=250, value=170)
weight = st.number_input("Enter your weight (kg):", min_value=30, max_value=200, value=70)
age = st.number_input("Enter your age:", min_value=5, max_value=100, value=25)

# Prediction Button
if st.button("Recommend Size"):
    features = np.array([[height, weight, age]])  # Modify based on model input format
    prediction = model.predict(features)[0]  # Assuming model returns a size label

    st.success(f"✅ Recommended Size: {prediction}")
