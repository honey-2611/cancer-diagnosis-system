# app.py
import streamlit as st
import joblib
import numpy as np
from fpdf import FPDF
import os

# ------------------ Page Configuration ------------------
st.set_page_config(page_title="Multi-Cancer Diagnosis System",
                   layout="wide")


st.markdown(
    """
    <style>
    .stMain { !important;
        background: #6242c2 !important;
        color: #e6ff26 !important;
    }
    .stAppHeader{
        background: black !important;
        color:#29d923 !important;
    }
    section[data-testid="stSidebar"] { 
        background: #dfe6aa !important;  
        color: red !important;
    }
    .stMarkdownContainer p{
    color: red !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Custom CSS Styling ------------------
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #e0b3ff, #c299ff);
        }
        section.main > div {
            background: linear-gradient(to right, #e0b3ff, #c299ff);
            color: black;
        }
        .stSelectbox div div div span {
            color: blue !important;
        }
        .stButton>button {
            color: white !important;
            background-color: #6a0dad;
            border-radius: 12px;
            padding: 10px 24px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ Sidebar ------------------
st.sidebar.markdown("""
    <h3 style='color:black;'>Select Cancer Type</h3>
""", unsafe_allow_html=True)

cancer_type = st.sidebar.selectbox("", ["Breast Cancer", "Lung Cancer", "Skin Cancer"])

st.sidebar.markdown("""
    <h4 style='color:black;'>About Project 🎗️</h4>
    <p style='color:black;'>This system helps detect Breast, Lung, and Skin cancers early using Machine Learning.</p>
    <p style='color:#ff2635;'>Made with ❤️ by <br>
    <strong>Honey Panchal <br> 
    Jaymin Patel <br>
    Dhruv Patel </strong> </p>
   
""", unsafe_allow_html=True)

# ------------------ Load Models ------------------
def load_model(cancer_type):
    models = {
        "Breast Cancer": "models/breast_cancer_model.pkl",
        "Lung Cancer": "models/lung_cancer_model.pkl",
        "Skin Cancer": "models/skin_cancer_model.pkl"
    }
    return joblib.load(models[cancer_type])

# ------------------ Seriousness Function ------------------
def seriousness_level(prob):
    if prob >= 0.85:
        return "High"
    elif prob >= 0.5:
        return "Moderate"
    else:
        return "Low"

# ------------------ PDF Generator ------------------
def generate_pdf(cancer_type, inputs, result, probability, seriousness):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt=f"{cancer_type} Diagnosis Report", ln=True, align='C')
    pdf.ln(10)

    for key, val in inputs.items():
        pdf.cell(200, 10, txt=f"{key}: {val}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Diagnosis Result: {result}", ln=True)
    pdf.cell(200, 10, txt=f"Probability: {probability:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Seriousness Level: {seriousness}", ln=True)

    pdf_path = f"report_{cancer_type.replace(' ', '_')}.pdf"
    pdf.output(pdf_path)
    return pdf_path

# ------------------ Input Forms ------------------
def get_inputs(cancer_type):
    inputs = {}
    if cancer_type == "Breast Cancer":
        features = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean',
            'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
            'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
            'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
            'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
            'concavity_worst', 'concave_points_worst', 'symmetry_worst',
            'fractal_dimension_worst']
        for feature in features:
            inputs[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=0.0, value=0.0)

    elif cancer_type == "Lung Cancer":
        options = ["Gender", "Smoking", "Yellow Fingers", "Anxiety", "Peer Pressure", "Chronic Disease", "Fatigue", "Allergy", "Wheezing", "Alcohol Consuming", "Coughing", "Shortness of Breath", "Swallowing Difficulty", "Chest Pain", "Lung cancer"]
        for opt in options:
            if opt == "Gender":
                inputs[opt] = st.selectbox(f"{opt}", ["Male", "Female", "Other"])
            else:
                inputs[opt] = st.selectbox(f"{opt}", ["Yes", "No"])

    elif cancer_type == "Skin Cancer":
        options = ["Gender", "Itching", "Ulcers", "Bleeding", "Elevation", "Age"]
        for opt in options:
            if opt == "Age":
                inputs[opt] = st.slider("Age", min_value=1, max_value=100, value=25)
            elif opt == "Gender":
                inputs[opt] = st.selectbox(f"{opt}", ["Male", "Female", "Other"])
            else:
                inputs[opt] = st.selectbox(f"{opt}", ["Yes", "No"])
    return inputs

# ------------------ Prediction ------------------
def preprocess_inputs(inputs, cancer_type):
    processed = []
    for val in inputs.values():
        if isinstance(val, str):
            processed.append(1 if val == "Yes" else 0)
        else:
            processed.append(val)
    return np.array(processed).reshape(1, -1)

# ------------------ Main ------------------
st.title("🎗️ Multi-Cancer Diagnosis System 🎗️")
st.markdown("<h3 style='color:#edc9f0;'>💗Early Detection Saves Lives</h3>", unsafe_allow_html=True)

inputs = get_inputs(cancer_type)

if st.button("🔎 Diagnose Now"):
    model = load_model(cancer_type)
    features = preprocess_inputs(inputs, cancer_type)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    result = "Positive" if prediction == 1 else "Negative"
    seriousness = seriousness_level(probability)

    st.success(f"Diagnosis Result: {result}")
    st.info(f"Probability: {probability:.2f}")
    st.warning(f"Seriousness Level: {seriousness}")

    # Generate and download PDF
    pdf_path = generate_pdf(cancer_type, inputs, result, probability, seriousness)
    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download Report as PDF", f, file_name=pdf_path)

    os.remove(pdf_path)
