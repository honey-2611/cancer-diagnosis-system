import streamlit as st
import joblib
import numpy as np
import os

# ✨ Set page config
st.set_page_config(page_title="Multi-Cancer Diagnosis System 🎗️", page_icon="🎗️", layout="wide")

# ✨ Apply Custom CSS Styling (Perfect Sidebar and Main Section Colors)
st.markdown("""
    <style>
    /* Main background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e0c3fc, #8ec5fc);
        backdrop-filter: blur(8px);
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #c084fc, #a855f7);
        border-radius: 10px;
        padding: 1rem;
    }

    /* Sidebar Cancer Type SelectBox text - WHITE */
    section[data-testid="stSidebar"] .stSelectbox div {
        color: white !important;
        font-weight: bold;
    }

    /* About Project and other sidebar texts - BLACK */
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] h3 {
        color: black !important;
    }

    /* Titles and Headings */
    h1, h2, h3 {
        color: #1e3a8a;
        font-weight: bold;
    }

    /* Normal Text */
    p, label, div {
        color: black;
        font-size: 18px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #facc15;
        color: white;
        border-radius: 12px;
        font-size: 18px;
        height: 3em;
        width: 15em;
        border: none;
    }

    /* Main Section Dropdown Text (Gender, Smoking etc) - YELLOW */
    div[data-baseweb="select"] span {
        color: #facc15 !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 📋 Sidebar Cancer Type Selection
cancer_type = st.sidebar.selectbox(
    "Select Cancer Type",
    ("Breast Cancer", "Lung Cancer", "Skin Cancer")
)

# 🚀 Load the correct model
def load_model(cancer_type):
    base_path = os.path.dirname(os.path.dirname(__file__))  # Go to root project folder
    models_path = os.path.join(base_path, 'models')

    if cancer_type == "Breast Cancer":
        model_file = os.path.join(models_path, 'breast_cancer_model.pkl')
    elif cancer_type == "Lung Cancer":
        model_file = os.path.join(models_path, 'lung_cancer_model.pkl')
    elif cancer_type == "Skin Cancer":
        model_file = os.path.join(models_path, 'skin_cancer_model.pkl')
    else:
        model_file = None

    if model_file and os.path.exists(model_file):
        model = joblib.load(model_file)
        return model
    else:
        st.error(f"❌ Model file not found: {model_file}")
        return None

model = load_model(cancer_type)

# 🏥 Title and Welcome Section
st.title("🎗️ Multi-Cancer Diagnosis System 🎗️")
st.markdown("<h3 style='text-align: center; color: #10b981;'>🏥 Early Detection Saves Lives</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Select a cancer type from the sidebar and enter patient details below. Our AI system will predict and assist you. 💖</p>", unsafe_allow_html=True)
st.markdown("---")

# 📋 Sidebar Extra Information
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920023.png", width=120)
    st.markdown("### About Project 🎗️")
    st.write("This system helps detect Breast, Lung, and Skin cancers early using Machine Learning.")
    st.markdown("""
    <hr style="border:1px solid white">
    Made with 💖 by <br> 
    <b>Honey Panchal</b> <br>
    <b>Jaymin Patel</b> <br> 
    <b>Dhruv Patel</b>
    """, unsafe_allow_html=True)

# 📝 Dynamic Input Fields
def get_inputs(cancer_type):
    inputs = []
    
    if cancer_type == "Breast Cancer":
        st.subheader("Enter Breast Cancer Patient Details")

        breast_features = [
            "Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", "Smoothness Mean",
            "Compactness Mean", "Concavity Mean", "Concave Points Mean", "Symmetry Mean", "Fractal Dimension Mean",
            "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE",
            "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE",
            "Radius Worst", "Texture Worst", "Perimeter Worst", "Area Worst", "Smoothness Worst",
            "Compactness Worst", "Concavity Worst", "Concave Points Worst", "Symmetry Worst", "Fractal Dimension Worst"
        ]

        for feature_name in breast_features:
            val = st.number_input(f"{feature_name}", step=0.01)
            inputs.append(val)

    elif cancer_type == "Lung Cancer":
        st.subheader("Enter Lung Cancer Patient Details")
        age = st.number_input("Age", 0, 120)
        gender = st.selectbox("Gender", ["Male", "Female"])
        smoking = st.selectbox("Smoking", ["Yes", "No"])
        yellow_fingers = st.selectbox("Yellow Fingers", ["Yes", "No"])
        anxiety = st.selectbox("Anxiety", ["Yes", "No"])
        peer_pressure = st.selectbox("Peer Pressure", ["Yes", "No"])
        chronic_disease = st.selectbox("Chronic Disease", ["Yes", "No"])
        fatigue = st.selectbox("Fatigue", ["Yes", "No"])
        allergy = st.selectbox("Allergy", ["Yes", "No"])
        wheezing = st.selectbox("Wheezing", ["Yes", "No"])
        alcohol_consuming = st.selectbox("Alcohol Consuming", ["Yes", "No"])
        coughing = st.selectbox("Coughing", ["Yes", "No"])
        shortness_of_breath = st.selectbox("Shortness of Breath", ["Yes", "No"])
        swallowing_difficulty = st.selectbox("Swallowing Difficulty", ["Yes", "No"])
        chest_pain = st.selectbox("Chest Pain", ["Yes", "No"])

        mapping = {"Male": 1, "Female": 0, "Yes": 1, "No": 0}
        inputs = [
            age,
            mapping[gender],
            mapping[smoking],
            mapping[yellow_fingers],
            mapping[anxiety],
            mapping[peer_pressure],
            mapping[chronic_disease],
            mapping[fatigue],
            mapping[allergy],
            mapping[wheezing],
            mapping[alcohol_consuming],
            mapping[coughing],
            mapping[shortness_of_breath],
            mapping[swallowing_difficulty],
            mapping[chest_pain]
        ]

    elif cancer_type == "Skin Cancer":
        st.subheader("Enter Skin Cancer Patient Details")
        age = st.number_input("Age", 0, 120)
        gender = st.selectbox("Gender", ["Male", "Female"])
        family_history = st.selectbox("Family History of Cancer", ["Yes", "No"])
        sun_exposure = st.selectbox("Sun Exposure", ["High", "Medium", "Low"])
        mole_growth = st.selectbox("Mole Growth", ["Yes", "No"])
        skin_pigmentation = st.selectbox("Skin Pigmentation", ["Dark", "Fair"])

        mapping = {"Male": 1, "Female": 0, "Yes": 1, "No": 0, "High": 2, "Medium": 1, "Low": 0, "Dark": 1, "Fair": 0}
        inputs = [
            age,
            mapping[gender],
            mapping[family_history],
            mapping[sun_exposure],
            mapping[mole_growth],
            mapping[skin_pigmentation]
        ]

    return np.array(inputs).reshape(1, -1)

features = get_inputs(cancer_type)

# 🧪 Predict on button click
from fpdf import FPDF
from datetime import datetime

# 🧪 Predict on button click
from fpdf import FPDF
from datetime import datetime

# 🧪 Predict on button click
if st.button("Diagnose Now 🚀"):
    if model is not None:
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]  # Probability of being 'Malignant'
        seriousness = ""

        # Determine seriousness based on probability
        if probability < 0.4:
            seriousness = "Mild"
        elif 0.4 <= probability < 0.7:
            seriousness = "Moderate"
        else:
            seriousness = "Severe"

        # Diagnosis Result Text
        diagnosis_result = "Malignant Cancer Detected" if prediction == 1 else "Benign Condition Detected"

        # Show result
        if prediction == 1:
            st.error(f"⚠️ Diagnosis: {diagnosis_result}")
            st.warning(f"🧪 Seriousness Level: {seriousness}")
        else:
            st.success(f"✅ Diagnosis: {diagnosis_result}")

        # Generate PDF Report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Cancer Diagnosis Report", ln=True, align="C")

        pdf.ln(10)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Cancer Type: {cancer_type}", ln=True)
        pdf.cell(0, 10, f"Diagnosis Result: {diagnosis_result}", ln=True)

        if prediction == 1:
            pdf.cell(0, 10, f"Seriousness Level: {seriousness}", ln=True)

        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", ln=True)

        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Patient Inputs:", ln=True)
        pdf.set_font("Arial", "", 12)

        for i, val in enumerate(features.flatten(), 1):
            pdf.cell(0, 10, f"Feature {i}: {val}", ln=True)

        # Save PDF
        pdf_output = "diagnosis_report.pdf"
        pdf.output(pdf_output)

        # Download button
        with open(pdf_output, "rb") as f:
            st.download_button(
                label="📄 Download Diagnosis Report",
                data=f,
                file_name="diagnosis_report.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("⚠️ Model loading failed.")
