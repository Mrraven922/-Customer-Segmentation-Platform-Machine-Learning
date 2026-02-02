import streamlit as st
import pandas as pd
import joblib

# ----------------------
# Load saved scaler and KMeans model
# ----------------------
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans_model.pkl")

# ----------------------
# App title
# ----------------------
st.title("Customer Segmentation App")
st.write("Enter customer details to predict the customer segment.")

# ----------------------
# User Inputs
# ----------------------
age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
total_spending = st.number_input("Total Spending (sum of purchases)", min_value=0, max_value=500000, value=5000)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=5)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=3)
num_web_visits = st.number_input("Number of Web Visits per Month", min_value=0, max_value=100, value=10)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

# ----------------------
# Create input dataframe
# ----------------------
# Create input dataframe
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})

# Rename columns to match training data
input_data = input_data.rename(columns={
    "Total Spending": "TotalSpending"
})

# Scale input
input_scaled = scaler.transform(input_data)

# ----------------------
# Scale input
# ----------------------
try:
    input_scaled = scaler.transform(input_data)
except ValueError as e:
    st.error(f"Feature mismatch error: {e}")
    st.stop()

# ----------------------
# Predict cluster
# ----------------------
if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    st.success(f"Predicted Segment: Cluster {cluster}")
