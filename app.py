import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_lottie import st_lottie
import requests

# Set page config for a professional look
st.set_page_config(page_title="Mental Health Predictor", page_icon="🧠", layout="centered")

# Custom CSS for animations and styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Assets
lottie_brain = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_07as9pvn.json")

# Load the Model [cite: 1, 3]
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Header Section
st_lottie(lottie_brain, height=200, key="coding")
st.title("Sentiment & Well-being Analyzer")
st.write("Enter the details below to predict the mental health category.")

# Input Form
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=20)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        academic_level = st.selectbox("Academic Level", ["Undergraduate", "Graduate", "High School", "Other"])
        country = st.text_input("Country", "USA")
        avg_usage = st.slider("Avg Daily Usage Hours", 0.0, 24.0, 5.0)

    with col2:
        platform = st.selectbox("Most Used Platform", ["Instagram", "TikTok", "YouTube", "Twitter", "Facebook", "Other"])
        academic_impact = st.selectbox("Affects Academic Performance", ["Yes", "No", "Maybe"])
        sleep_hours = st.slider("Sleep Hours Per Night", 0.0, 12.0, 7.0)
        mh_score = st.slider("Mental Health Score", 0.0, 10.0, 5.0)

# Prediction Logic
if st.button("Analyze Results"):
    # Prepare the input features matching model.pkl 
    # Note: Categorical variables (Gender, Country, etc.) may need encoding 
    # based on how you trained the model. This example assumes standard passing.
    
    features = np.array([[age, gender, academic_level, country, avg_usage, 
                         platform, academic_impact, sleep_hours, mh_score]])
    
    # Simple check: Ensure categorical data is handled if your model expects numbers
    # For this snippet, we'll display the prediction result directly
    try:
        prediction = model.predict(features)
        
        st.divider()
        st.subheader("Analysis Result:")
        
        result = prediction[0]
        if result == "Positive":
            st.success(f"The predicted state is: **{result}** ✨")
        elif result == "Neutral":
            st.info(f"The predicted state is: **{result}** ⚖️")
        else:
            st.warning(f"The predicted state is: **{result}** ⚠️")
            
    except Exception as e:
        st.error("Encoding Error: Ensure categorical inputs match the model's training format.")
