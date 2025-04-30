# 🎗️ Multi-Cancer Diagnosis System

An AI-powered web application built using **Streamlit** that diagnoses **Breast Cancer**, **Lung Cancer**, and **Skin Cancer** using 
trained machine learning models. The system also provides a seriousness level of the condition and generates a downloadable PDF report of the results.

---

## 🚀 Live Demo

👉 [Click here to access the live app](https://your-app-url.streamlit.app)

---

## 🧠 Features

✅ Diagnose three major types of cancer  
✅ Machine learning models with trained accuracy  
✅ Seriousness level output (Low / Moderate / High)  
✅ Downloadable PDF report of diagnosis  
✅ Beautiful and intuitive user interface  
✅ Fully deployable with Streamlit Cloud

---

## 📁 Folder Structure

Multi-Cancer-Diagnosis-System/ 
├── app.py 
├── models/ 
│ ├── breast_cancer_model.pkl 
│ ├── lung_cancer_model.pkl 
│ └── skin_cancer_model.pkl 
├── datasets/ 
│ ├── breast_cancer.csv 
│ ├── lung_cancer.csv 
│ └── skin_cancer.csv 
├── requirements.txt 
└── README.md


---

## 📦 Installation (for local use)

Make sure you have Python 3.8+ installed.

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
streamlit run app.py


 Technologies Used
Python
Streamlit
scikit-learn
XGBoost
Pandas / NumPy
FPDF (for PDF generation)

🧬 Machine Learning Models

Each model was trained using its respective dataset:
Breast Cancer: Logistic Regression
Lung Cancer: Random Forest
Skin Cancer: Random Forest / XGBoost

Models were evaluated using:

Accuracy Score
Classification Report
Confusion Matrix

📄 PDF Reports

After prediction, users can download a PDF report that includes:
User input values
Prediction result
Probability score
Seriousness level

👨‍💻 Contributors

Honey Panchal
Jaymin Patel
Dhruv Patel

🌟 Acknowledgements

Special thanks to:
Streamlit for making app deployment so simple
Kaggle for public cancer datasets
Our mentors and teammates for constant support

📬 Contact
For any inquiries or feedback, feel free to reach out via GitHub Issues or connect on LinkedIn.
