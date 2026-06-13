import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import time

# --- 1. Page Configuration ---
st.set_page_config(page_title="AI Fraud Detection System", page_icon="🛡️", layout="centered")

# --- 2. Load Model and Data (Cached for Speed) ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('fraud_detection_model.h5')

@st.cache_data
def load_data():
    # Load a small sample to simulate live transactions
    df = pd.read_csv('sample_data.csv').sample(2000, random_state=42)
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    return df

model = load_model()
df = load_data()

# Pre-fit the scaler on the sample amount data
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# --- 3. UI Dashboard Layout ---
st.title("🛡️ Neural Network Fraud Detection")
st.markdown("### MBA Data Science Major Project")
st.write("This dashboard simulates real-time transaction monitoring using a Deep Learning model. Click the button below to pull a random transaction from the network and evaluate its behavioral pattern.")

st.divider()

# --- 4. Simulation Logic ---
if st.button("🔄 Intercept & Analyze Random Transaction", use_container_width=True):
    with st.spinner("Analyzing behavioral patterns and network nodes..."):
        time.sleep(1) # Artificial delay for effect
        
        # Pick a random row
        sample_tx = df.sample(1)
        actual_class = sample_tx['Class'].values[0]
        amount_usd = sample_tx['Amount'].values[0]
        
        # Prepare features for the model
        features = sample_tx.drop(['Class', 'Amount', 'Amount_Scaled'], axis=1, errors='ignore')
        # Insert the scaled amount at the end to match training shape
        features['Amount'] = sample_tx['Amount_Scaled'].values[0]
        
        # Run prediction
        prediction_prob = model.predict(features, verbose=0)[0][0]
        is_fraud = prediction_prob > 0.5
        
        # --- 5. Display Results ---
        st.subheader("Transaction Intercepted:")
        col1, col2, col3 = st.columns(3)
        col1.metric("Transaction Amount", f"${amount_usd:,.2f}")
        col2.metric("AI Confidence Score", f"{prediction_prob * 100:.2f}%")
        col3.metric("True Label (Hidden)", "Fraud" if actual_class == 1 else "Legitimate")
        
        st.markdown("### AI System Verdict:")
        if is_fraud:
            st.error("🚨 **ALERT: FRAUDULENT PATTERN DETECTED.** Transaction Blocked.")
        else:
            st.success("✅ **CLEARED:** Transaction behavior is normal. Payment Approved.")
        
        st.write("---")
        st.write("**Raw Behavioral Features (PCA Vectors V1-V28):**")
        st.dataframe(features.style.format("{:.4f}"))