import streamlit as st
import numpy as np
import pickle

# -------------------------------------------
# PAGE CONFIG
# -------------------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="centered")

# -------------------------------------------
# LOAD MODEL
# -------------------------------------------
try:
    kmeans = pickle.load(open("kmeans.pkl", "rb"))
except FileNotFoundError:
    st.error("❌ Model file 'kmeans.pkl' not found! Please place it in the same folder as this app.")
    st.stop()

# -------------------------------------------
# STYLING (CUSTOM CSS)
# -------------------------------------------
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #ff7e5f, #feb47b);
        color: white;
    }
    .main {
        background: rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0px 4px 25px rgba(0,0,0,0.3);
    }
    h1 { text-align:center; }
</style>
""", unsafe_allow_html=True)
# -------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------
def clustering(age, avg_spend, visit_per_week, promotion_interest):
    new_customer = np.array([[age, avg_spend, visit_per_week, promotion_interest]])
    predicted_cluster = kmeans.predict(new_customer)

    if predicted_cluster[0] == 0:
        return "🛒 Daily"
    elif predicted_cluster[0] == 1:
        return "🎉 Weekend"
    else:
        return "💎 Promotion"

# -------------------------------------------
# UI CONTENT
# -------------------------------------------
st.markdown("<h1>🧠 Customer Segmentation App</h1>", unsafe_allow_html=True)
st.write("### Use this tool to predict which **customer group** a person belongs to!")

st.markdown("<div class='main'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("👤 Age", min_value=18, max_value=100, value=30)
    visit_per_week = st.number_input("🕒 Visits per Week", min_value=0, max_value=20, value=5)
with col2:
    avg_spend = st.number_input("💰 Average Spend ($)", min_value=0.0, max_value=1000.0, value=45.0)
    promotion_interest = st.number_input("🎯 Promotion Interest (1-10)", min_value=0, max_value=10, value=7)

if st.button("🔍 Predict Cluster"):
    cluster_label = clustering(age, avg_spend, visit_per_week, promotion_interest)
    st.markdown(f"<div class='result-box'>✅ The customer belongs to the <b>{cluster_label}</b> cluster.</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
