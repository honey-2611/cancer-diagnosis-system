import streamlit as st
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Multi-Cancer Diagnosis System", layout="wide")

# ✨ Custom Styling - Navbar and Theme
st.markdown("""
    <style>
    /* Change the background color */
    body {
        background: linear-gradient(to right, #dda0dd, #e0c3fc); /* Light purple gradient */
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #d8bfd8, #dda0dd); /* Slightly darker purple */
    }
    [data-testid="stSidebar"] .css-ng1t4o {
        color: white; /* Sidebar Text */
    }
    [data-testid="stSidebar"] .css-1cpxqw2 {
        color: white; /* Sidebar Select Text */
    }

    /* Change the top navbar text color */
    header[data-testid="stHeader"] {
        background-color: black;
    }
    header[data-testid="stHeader"] div[class^='st-emotion-cache'] {
        color: #2ed573; /* Sea Green color */
    }

    /* Buttons and selection text color */
    .stButton>button {
        color: white;
        background-color: #6a0dad;
        font-weight: bold;
    }
    .stSelectbox>div>div>div>div {
        color: yellow;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Models
def load_model(cancer_type):
    if cancer_type == "Breast Cancer":
        return joblib.load('models/breast_cancer_model.pkl')
    elif cancer_type == "Lung Cancer":
        return joblib.load('models/lung_cancer_model.pkl')
    elif cancer_type == "Skin Cancer":
        return joblib.load('models/skin_cancer_model.pkl')

# Sidebar
st.sidebar.title("Select Cancer Type")
cancer_type = st.sidebar.selectbox("", ["Breast Cancer", "Lung Cancer", "Skin Cancer"])

# Main Title
st.title("🎗️ Multi-Cancer Diagnosis System 🎗️")
st.subheader("🌸 Early Detection Saves Lives 🌸")

# Input fields based on cancer type
def get_input_fields(cancer_type):
    if cancer_type == "Breast Cancer":
        fields = [
            "Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean",
            "Smoothness Mean", "Compactness Mean", "Concavity Mean", "Concave Points Mean",
            "Symmetry Mean", "Fractal Dimension Mean", "Radius SE", "Texture SE",
            "Perimeter SE", "Area SE", "Smoothness SE", "Compactness SE",
            "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE",
            "Radius Worst", "Texture Worst", "Perimeter Worst", "Area Worst",
            "Smoothness Worst", "Compactness Worst", "Concavity Worst",
            "Concave Points Worst", "Symmetry Worst", "Fractal Dimension Worst"
        ]
    elif cancer_type == "Lung Cancer":
        fields = [
            "Gender", "Age", "Smoking", "Yellow Fingers", "Anxiety",
            "Peer Pressure", "Chronic Disease", "Fatigue", "Allergy", "Wheezing",
            "Alcohol Consuming", "Coughing", "Shortness of Breath",
            "Swallowing Difficulty", "Chest Pain"
        ]
    elif cancer_type == "Skin Cancer":
        fields = [
            "Age", "Gender", "Sun Exposure", "Family History", "Number of Moles",
            "Skin Type", "Freckles", "Tanning Ability", "Genetic Risk", "Immune Status"
        ]
    return fields

fields = get_input_fields(cancer_type)

st.header(f"Enter {cancer_type} Patient Details 👩‍⚕️👨‍⚕️")
input_data = []

# Input form
for field in fields:
    value = st.number_input(f"{field}", step=0.01, format="%.2f")
    input_data.append(value)

input_array = np.array(input_data).reshape(1, -1)

# Predict button
if st.button("🔎 Diagnose Now"):
    model = load_model(cancer_type)
    prediction = model.predict(input_array)

    if prediction[0] == 1:
        st.success(f"🔴 Warning: {cancer_type} Detected!")
    else:
        st.success(f"🟢 Good News: No {cancer_type} Detected!")

# Sidebar - About Project
st.sidebar.title("About Project 🎗️")
st.sidebar.write("""
This system helps detect Breast, Lung, and Skin cancers early using Machine Learning.

Made with 💖 by  
Honey Panchal,  
Jaymin Patel,  
Dhruv Patel.
""")
