import streamlit as st
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Multi-Cancer Diagnosis System", layout="wide")

# Inject custom CSS for background and styling
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(to right, #b171f0, #953bed);
    }

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #b8b3fc, #9892e8);
    }

    /* Dropdown and text color */
    .stSelectbox > div[data-baseweb="select"] > div {
        color: white !important;
    }

    /* Label text color */
    label, .st-bb, .st-c3 {
        color: yellow !important;
        font-weight: 600;
    }

    /* Share menu and top bar */
    header { color: #39FF14 !important; }

    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.markdown("<h4 style='color:#121112;'>Select Cancer Type</h4>", unsafe_allow_html=True)
    cancer_type = st.selectbox("", ["Breast Cancer", "Lung Cancer", "Skin Cancer"])

    st.markdown("""
    <h4 style='color:#121112;'>About Project 🎗️</h4>
    <p style='color:#121112;'>
    This system helps detect Breast, Lung, and Skin cancers early using Machine Learning.
    <br><br>
    Made with 💖 by<br>
    <h4 span style='color:#121112;'>Honey Panchal,<br>
    Jaymin Patel,<br>
    Dhruv Patel.</span></h4>
    </p>
    """, unsafe_allow_html=True)

# Function to load model
@st.cache_data

def load_model(cancer_type):
    model_paths = {
        "Breast Cancer": "models/breast_cancer_model.pkl",
        "Lung Cancer": "models/lung_cancer_model.pkl",
        "Skin Cancer": "models/skin_cancer_model.pkl"
    }
    model_file = model_paths.get(cancer_type)
    if model_file and os.path.exists(model_file):
        return joblib.load(model_file)
    else:
        st.error(f"❌ Model file not found: {model_file}")
        return None

# Get feature input based on cancer type
def get_input_fields(cancer_type):
    if cancer_type == "Breast Cancer":
        features = [
            "Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", "Smoothness Mean",
            "Compactness Mean", "Concavity Mean", "Concave Points Mean", "Symmetry Mean", "Fractal Dimension Mean",
            "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE",
            "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE",
            "Radius Worst", "Texture Worst", "Perimeter Worst", "Area Worst", "Smoothness Worst",
            "Compactness Worst", "Concavity Worst", "Concave Points Worst", "Symmetry Worst", "Fractal Dimension Worst"
        ]
        inputs = [st.number_input(label, min_value=0.0, step=0.01, format="%.2f") for label in features]
        return np.array(inputs).reshape(1, -1)

    elif cancer_type == "Lung Cancer":
        fields = [
            ("Gender", ["Male", "Female"]),
            ("Smoking", ["Yes", "No"]),
            ("Yellow Fingers", ["Yes", "No"]),
            ("Anxiety", ["Yes", "No"]),
            ("Peer Pressure", ["Yes", "No"]),
            ("Chronic Disease", ["Yes", "No"]),
            ("Fatigue", ["Yes", "No"]),
            ("Allergy", ["Yes", "No"]),
            ("Wheezing", ["Yes", "No"]),
            ("Alcohol Consuming", ["Yes", "No"]),
            ("Coughing", ["Yes", "No"]),
            ("Shortness of Breath", ["Yes", "No"]),
            ("Swallowing Difficulty", ["Yes", "No"]),
            ("Chest Pain", ["Yes", "No"]),
            ("Age", list(range(1, 101)))
        ]
        input_values = []
        for label, options in fields:
            if options[0] in ["Yes", "No"]:
                val = st.selectbox(label, options)
                input_values.append(1 if val == "Yes" else 2)
            elif label == "Gender":
                val = st.selectbox(label, options)
                input_values.append(1 if val == "Male" else 0)
            else:
                val = st.selectbox(label, options)
                input_values.append(val)
        return np.array(input_values).reshape(1, -1)

    elif cancer_type == "Skin Cancer":
        fields = ["Age", "Itching", "Irritation", "Ulcer", "Bleeding", "Swelling", "Pain"]
        return np.array([st.number_input(field, min_value=0.0, step=1.0) for field in fields]).reshape(1, -1)

# Main content
st.markdown(f"""
    <h1 style='color:#121112;'>🎗️ Multi-Cancer Diagnosis System 🎗️</h1>
    <h3 style='color:#efebf2;'>🌸 Early Detection Saves Lives 🌸</h3>
    <h2 style='color:#121112;'>Enter {cancer_type} Patient Details 👩‍⚕️👨‍⚕️</h2>
""", unsafe_allow_html=True)

features = get_input_fields(cancer_type)
model = load_model(cancer_type)

if st.button("Diagnose Now", key="diagnose_btn"):
    if model is not None:
        prediction = model.predict(features)[0]
        result = "🎉 No Cancer Detected!" if prediction == 0 else "⚠️ Cancer Detected - Please consult a doctor!"
        st.success(result)
