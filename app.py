import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Alzheimer's Disease Classifier", layout="wide")

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------

def load_pickle_safe(path):
    """Safely loads a pickle file and handles errors."""
    if not os.path.exists(path):
        st.error(f"Missing file: {path}. Add it to repo root or upload via sidebar.")
        st.stop()
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        st.stop()


def predict_and_proba(model, scaler, input_dict, numeric_cols, features_order):
    """
    Handles scaler-transform robustness:
    - Matches expected feature names
    - Fills missing columns with zeros
    - Falls back to array transform if needed
    """
    X = pd.DataFrame([input_dict])

    # Ensure all required features exist
    for f in features_order:
        if f not in X.columns:
            X[f] = 0

    X = X[features_order]  # reorder correctly

    # Try scaling robustly
    try:
        if hasattr(scaler, "feature_names_in_"):
            expected = list(scaler.feature_names_in_)

            # Build DataFrame with expected columns
            X_exp = pd.DataFrame(columns=expected, index=[0])
            for col in expected:
                X_exp.at[0, col] = X[col] if col in X.columns else 0

            # Convert numerical safely
            for col in X_exp.columns:
                try:
                    X_exp[col] = pd.to_numeric(X_exp[col])
                except:
                    pass

            scaled = scaler.transform(X_exp)
            X_scaled_df = pd.DataFrame(scaled, columns=expected)

            # Replace numeric columns
            for col in numeric_cols:
                if col in X_scaled_df:
                    X[col] = X_scaled_df[col]
        else:
            # Fallback: scaler was trained on arrays
            cols_to_scale = [c for c in numeric_cols if c in X.columns]
            X[cols_to_scale] = scaler.transform(X[cols_to_scale].values)

    except Exception as e1:
        st.error(f"Scaler transform failed: {e1}")
        st.stop()

    # Predict
    try:
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[:, 1][0] if hasattr(model, "predict_proba") else None
    except Exception as e2:
        st.error(f"Prediction failed: {e2}")
        st.stop()

    return pred, proba


def risk_label(prob):
    """Returns color-coded risk label."""
    if prob is None:
        return "Unknown", "gray"
    if prob < 0.3:
        return "Low Risk", "#4caf50"
    elif prob < 0.7:
        return "Moderate Risk", "#ffc107"
    else:
        return "High Risk", "#f44336"


# ---------------------------------------------------
# Load Model + Scaler
# ---------------------------------------------------

MODEL_PATH = "best_alzheimers_model.pkl"
SCALER_PATH = "scaler_alzheimers.pkl"

model_file = st.sidebar.file_uploader("Upload best_alzheimers_model.pkl", type=["pkl"])
scaler_file = st.sidebar.file_uploader("Upload scaler_alzheimers.pkl", type=["pkl"])

if model_file and scaler_file:
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    st.sidebar.success("Loaded model & scaler from uploads.")
else:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = load_pickle_safe(MODEL_PATH)
        scaler = load_pickle_safe(SCALER_PATH)
    else:
        st.sidebar.error("Upload model & scaler or add them to repo.")
        st.stop()

# ---------------------------------------------------
# UI
# ---------------------------------------------------

st.title("🧠 Alzheimer’s Disease Risk Classifier")
st.write("This app predicts Alzheimer’s probability using a trained ML model + StandardScaler.")

st.sidebar.header("Patient Input Features")

# ========== SIDEBAR INPUT ==========
age = st.sidebar.number_input("Age", 60, 100, 72)
gender = st.sidebar.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
ethnicity = st.sidebar.selectbox("Ethnicity", [0,1,2,3], format_func=lambda x: ["Caucasian","African American","Asian","Other"][x])
education = st.sidebar.selectbox("Education Level", [0,1,2,3], format_func=lambda x: ["None","High School","Bachelor's","Higher"][x])
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0)

smoking = st.sidebar.selectbox("Smoking", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
alcohol = st.sidebar.number_input("Alcohol Consumption (units/week)", 0.0, 30.0, 1.0)
physical = st.sidebar.number_input("Physical Activity (hrs/week)", 0.0, 20.0, 2.0)
diet = st.sidebar.slider("Diet Quality (0–10)", 0, 10, 6)
sleep = st.sidebar.slider("Sleep Quality (4–10)", 4, 10, 7)

family = st.sidebar.selectbox("Family History", [0,1])
cardio = st.sidebar.selectbox("Cardiovascular Disease", [0,1])
diabetes = st.sidebar.selectbox("Diabetes", [0,1])
depression = st.sidebar.selectbox("Depression", [0,1])
headinjury = st.sidebar.selectbox("Head Injury", [0,1])
hypertension = st.sidebar.selectbox("Hypertension", [0,1])

sbp = st.sidebar.number_input("Systolic BP", 80, 200, 130)
dbp = st.sidebar.number_input("Diastolic BP", 50, 130, 80)

chol_total = st.sidebar.number_input("Cholesterol Total", 100, 400, 200)
chol_ldl = st.sidebar.number_input("LDL", 20, 300, 120)
chol_hdl = st.sidebar.number_input("HDL", 10, 120, 50)
trig = st.sidebar.number_input("Triglycerides", 30, 600, 150)

mmse = st.sidebar.number_input("MMSE Score", 0, 30, 27)
functional = st.sidebar.number_input("Functional Assessment", 0, 10, 8)
adl = st.sidebar.number_input("ADL Score", 0, 10, 8)

memory = st.sidebar.selectbox("Memory Complaints", [0,1])
behavior = st.sidebar.selectbox("Behavioral Problems", [0,1])
conf = st.sidebar.selectbox("Confusion", [0,1])
disorient = st.sidebar.selectbox("Disorientation", [0,1])
personality = st.sidebar.selectbox("Personality Changes", [0,1])
tasks = st.sidebar.selectbox("Difficulty Completing Tasks", [0,1])
forget = st.sidebar.selectbox("Forgetfulness", [0,1])

# ---------------------------------------------------
# Input Dictionary (Order MUST match training)
# ---------------------------------------------------

features_order = [
    "Age","Gender","Ethnicity","EducationLevel","BMI","Smoking","AlcoholConsumption",
    "PhysicalActivity","DietQuality","SleepQuality","FamilyHistoryAlzheimers",
    "CardiovascularDisease","Diabetes","Depression","HeadInjury","Hypertension",
    "SystolicBP","DiastolicBP","CholesterolTotal","CholesterolLDL","CholesterolHDL",
    "CholesterolTriglycerides","MMSE","FunctionalAssessment","MemoryComplaints",
    "BehavioralProblems","ADL","Confusion","Disorientation","PersonalityChanges",
    "DifficultyCompletingTasks","Forgetfulness"
]

input_dict = {
    "Age": age,
    "Gender": gender,
    "Ethnicity": ethnicity,
    "EducationLevel": education,
    "BMI": bmi,
    "Smoking": smoking,
    "AlcoholConsumption": alcohol,
    "PhysicalActivity": physical,
    "DietQuality": diet,
    "SleepQuality": sleep,
    "FamilyHistoryAlzheimers": family,
    "CardiovascularDisease": cardio,
    "Diabetes": diabetes,
    "Depression": depression,
    "HeadInjury": headinjury,
    "Hypertension": hypertension,
    "SystolicBP": sbp,
    "DiastolicBP": dbp,
    "CholesterolTotal": chol_total,
    "CholesterolLDL": chol_ldl,
    "CholesterolHDL": chol_hdl,
    "CholesterolTriglycerides": trig,
    "MMSE": mmse,
    "FunctionalAssessment": functional,
    "MemoryComplaints": memory,
    "BehavioralProblems": behavior,
    "ADL": adl,
    "Confusion": conf,
    "Disorientation": disorient,
    "PersonalityChanges": personality,
    "DifficultyCompletingTasks": tasks,
    "Forgetfulness": forget
}

numeric_cols = [
    "Age","BMI","AlcoholConsumption","PhysicalActivity","DietQuality","SleepQuality",
    "SystolicBP","DiastolicBP","CholesterolTotal","CholesterolLDL","CholesterolHDL",
    "CholesterolTriglycerides","MMSE","FunctionalAssessment","ADL"
]

# ---------------------------------------------------
# Predict Button
# ---------------------------------------------------

if st.button("Predict Alzheimer’s Risk"):
    pred, proba = predict_and_proba(model, scaler, input_dict, numeric_cols, features_order)

    label, color = risk_label(proba)

    st.markdown(f"### Prediction: **{'Alzheimer’s' if pred==1 else 'No Alzheimer’s'}**")
    if proba is not None:
        st.markdown(f"### Risk Probability: `{proba:.3f}`")
    st.markdown(f"<div style='padding:10px;background:{color};color:white;border-radius:6px'>{label}</div>", unsafe_allow_html=True)

    st.subheader("Input Summary")
    st.dataframe(pd.DataFrame([input_dict]))

else:
    st.info("Fill the sidebar and click **Predict Alzheimer’s Risk**.")

# Footer
st.markdown("---")
st.caption("👩‍💻 Built by Pushpam Kumari | Alzheimer’s Disease ML Classifier | Streamlit Deployment")
