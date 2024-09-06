import streamlit as st
import pickle
import numpy as np


# Set custom page configuration
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Add background image or color (optional)
st.markdown(
    """
    <style>
    .main {
        background-color: #f4f4f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #5c6bc0;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #3949ab;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description with an icon
st.title("🍷 Wine Quality Prediction App")
st.markdown("Welcome to the Wine Quality Prediction App! Enter the wine features and predict the quality of the wine.")


# Define and display features with better visualization
st.subheader("🔍 Enter the Wine Features:")

# Input fields for the features
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.7)
citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.0)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=20.0, value=1.9)
chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, value=34.0)
density = st.number_input("Density", min_value=0.0, max_value=2.0, value=0.9978)
pH = st.number_input("pH", min_value=0.0, max_value=14.0, value=3.51)
sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.56)
alcohol = st.number_input("Alcohol", min_value=0.0, max_value=20.0, value=9.4)

# Button to make prediction
if st.button("Predict Wine Quality"):
    # Load the model
    try:
        with open('wine_prediction.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
            # Prepare input data
            input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                    chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                                    density, pH, sulphates, alcohol]])
            # Make prediction
            prediction = model.predict(input_data)
            if prediction[0] == 1:
                st.success("The predicted wine quality is: Good Quality 🍷")
            else:
                st.warning("The predicted wine quality is: Bad Quality 🚫")
    except FileNotFoundError:
        st.error("Model file 'wine_prediction.pkl' not found. Please ensure the file is in the correct location.")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")

# Add a footer with additional information or links (optional)
st.markdown(
    """
    <hr>
    <small>Developed by Gokul | 2024</small>
    """,
    unsafe_allow_html=True
)
