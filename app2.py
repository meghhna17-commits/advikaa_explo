import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load('model.pkl')

# Load dataset (use relative path)
df = pd.read_excel("mp_dataset_processed.xlsx")

# Features and targets
feature_names = list(df.columns[:10])
target_names = list(df.columns[-2:])

# Title
st.title("FormuGen AI: Predict. Optimize. Deliver")

st.markdown("### Enter input feature values:")

# Inputs
inputs = []
for col in feature_names:
    val = st.number_input(label=col, value=0.0)
    inputs.append(val)

# Prediction
if st.button("Predict"):
    data = np.array(inputs).reshape(1, -1)
    prediction = model.predict(data)

    st.subheader("Prediction Results:")
    
    for i in range(len(target_names)):
        st.metric(label=target_names[i], value=round(prediction[0][i], 4))
