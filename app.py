import streamlit as st
import joblib
import numpy as np
from pydantic import BaseModel

# Load the machine learning model
model = joblib.load('frauddetection.pkl')

# Define the Streamlit app
st.title("Credit Card Fraud Detection API")
st.markdown("""
An API that utilizes a Machine Learning model to detect if a credit card transaction is fraudulent or not based on the following features: hours, amount, transaction type etc.
""")

# Define the input form
st.sidebar.title("Input Data")
step = st.sidebar.number_input("Step", value=1)
types = st.sidebar.number_input("Type", value=1)
amount = st.sidebar.number_input("Amount", value=0.0)
oldbalanceorig = st.sidebar.number_input("Old Balance Orig", value=0.0)
newbalanceorig = st.sidebar.number_input("New Balance Orig", value=0.0)
oldbalancedest = st.sidebar.number_input("Old Balance Dest", value=0.0)
newbalancedest = st.sidebar.number_input("New Balance Dest", value=0.0)
isflaggedfraud = st.sidebar.number_input("Is Flagged Fraud", value=0.0)

# Define the prediction function
def predict(step, types, amount, oldbalanceorig, newbalanceorig, oldbalancedest, newbalancedest, isflaggedfraud):
    features = np.array([[step, types, amount, oldbalanceorig, newbalanceorig, oldbalancedest, newbalancedest, isflaggedfraud]])
    predictions = model.predict(features)
    if predictions == 1:
        return "Fraudulent"
    else:
        return "Not Fraudulent"

# Make predictions and display the result
if st.sidebar.button("Predict"):
    result = predict(step, types, amount, oldbalanceorig, newbalanceorig, oldbalancedest, newbalancedest, isflaggedfraud)
    st.write(f"Prediction: {result}")
