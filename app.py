import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Alzheimer's Disease Classifier", layout="wide")

# -----------------------
# Helper functions
# -----------------------
def load_pickle_safe(path):
    if not os.path.exists(path):
        st.error(f"Required file not found: {path}. Upload this file to the repo root or use the sidebar uploader.")
        st.stop()
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        st.stop()

def predict_and_proba(model, scaler, input_dict, numeric_cols, features_order):
    # Build dataframe with proper feature order (add missing features as 0)
    X = pd.DataFrame([input_dict])
    # ensure all features in features_order are present
    for f in features_order:
        if f not in X.columns:
            X[f] = 0
    X = X[features_order]  # reorder to match training
    
    # Scale numeric columns if scaler is provided
    cols_to_scale = [c for c in numeric_cols if c in X.columns]
    if len(cols_to_scale) > 0 and scaler is not None:
        try:
            X[cols_to_scale] = scaler.transform(X[cols_to_scale])
        except Exception as e:
            st.error(f"Scaler transform failed: {e}")
            st.stop()
    # predict
    try:
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[:, 1][0])
        else:
            proba = None
        pred = int(model.predict(X)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    return pred, proba

def risk_label(prob):
    if prob is None:
        return "Unknown", "gray"
    if prob < 0.3:
        return "Low risk", "#4caf50"
    elif prob < 0.7:
        return "Moderate risk", "#ffc107"
    else:
        return "High risk", "#f44336"

# -----------------------
# Try to load model and scaler from repo root
# -----------------------
MODEL_PATH = "best_alzheimers_model.pkl"
SCALER_PATH = "scaler_alzheimers.pkl"

uploaded_model = st.sidebar.file_uploader("Upload best_alzheimers_model.pkl", type=["pkl","joblib"])
uploaded_scaler = st.sidebar.file_uploader("Upload scaler_alzheimers.pkl", type=["pkl","joblib"])

if uploaded_model and uploaded_scaler:
    try:
        model = joblib.load(uploaded_model)
        scaler = joblib.load(uploaded_scaler)
        st.sidebar.success("Model & scaler loaded from uploads.")
    except Exception as e:
        st.sidebar.error("Uploaded files could not be loaded. See details below.")
        st.sidebar.text(str(e))
        st.stop()
else:
    # If not uploaded via UI, try loading from repo files
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = load_pickle_safe(MODEL_PATH)
        scaler = load_pickle_safe(SCALER_PATH)
    else:
        st.sidebar.info("Upload model & scaler via the sidebar, or add 'best_alzheimers_model.pkl' and 'scaler_alzheimers.pkl' to the repo root.")
        st.stop()

st.title("🧠 Alzheimer's Disease Risk Classifier")
st.write("Use the sidebar to input patient features. The app uses a pre-trained classifier and a StandardScaler to estimate Alzheimer's risk.")

# -----------------------
# Sidebar inputs (example defaults)
# -----------------------
st.sidebar.header("Patient Features (example defaults)")

age = st.sidebar.number_input("Age", min_value=60, max_value=100, value=72, step=1)
gender = st.sidebar.selectbox("Gender", options=[0,1], format_func=lambda x: "Male" if x==0 else "Female", index=1)
ethnicity = st.sidebar.selectbox("Ethnicity", options=[0,1,2,3], format_func=lambda x: ["Caucasian","African American","Asian","Other"][x])
education = st.sidebar.selectbox("Education Level", options=[0,1,2,3], format_func=lambda x: ["None","High School","Bachelor's","Higher"][x])
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
smoking = st.sidebar.selectbox("Smoking", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes")
alcohol = st.sidebar.number_input("Alcohol units/week", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
physical_activity = st.sidebar.number_input("Physical activity (hrs/week)", min_value=0.0, max_value=40.0, value=2.0, step=0.1)
diet_quality = st.sidebar.slider("Diet quality (0-10)", 0, 10, 6)
sleep_quality = st.sidebar.slider("Sleep quality (4-10)", 4, 10, 7)
family_hist = st.sidebar.selectbox("Family history of Alzheimer's", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes")
cardio = st.sidebar.selectbox("Cardiovascular disease", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes")
diabetes = st.sidebar.selectbox("Diabetes", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes")
depression = st.sidebar.selectbox("Depression", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes")
head_injury = st.sidebar.selectbox("Head injury history", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes")
hypertension = st.sidebar.selectbox("Hypertension", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes")

systolic = st.sidebar.number_input("Systolic BP", min_value=80, max_value=200, value=130, step=1)
diastolic = st.sidebar.number_input("Diastolic BP", min_value=50, max_value=130, value=80, step=1)
chol_total = st.sidebar.number_input("Cholesterol total", min_value=100, max_value=400, value=200, step=1)
chol_ldl = st.sidebar.number_input("Cholesterol LDL", min_value=20, max_value=300, value=120, step=1)
chol_hdl = st.sidebar.number_input("Cholesterol HDL", min_value=10, max_value=120, value=50, step=1)
triglycerides = st.sidebar.number_input("Triglycerides", min_value=30, max_value=600, value=150, step=1)

mmse = st.sidebar.number_input("MMSE (0-30)", min_value=0, max_value=30, value=27, step=1)
functional = st.sidebar.number_input("Functional Assessment (0-10)", min_value=0, max_value=10, value=8, step=1)
adl = st.sidebar.number_input("ADL (0-10)", min_value=0, max_value=10, value=8, step=1)

memory_complaints = st.sidebar.selectbox("Memory complaints", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes")
behavioral = st.sidebar.selectbox("Behavioral problems", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes")
confusion = st.sidebar.selectbox("Confusion", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes")
disorientation = st.sidebar.selectbox("Disorientation", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes")
personality_changes = st.sidebar.selectbox("Personality changes", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes")
difficulty_tasks = st.sidebar.selectbox("Difficulty completing tasks", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes")
forgetfulness = st.sidebar.selectbox("Forgetfulness", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes")

# feature order must match training
features_order = [
    "Age","Gender","Ethnicity","EducationLevel","BMI","Smoking","AlcoholConsumption",
    "PhysicalActivity","DietQuality","SleepQuality","FamilyHistoryAlzheimers","CardiovascularDisease",
    "Diabetes","Depression","HeadInjury","Hypertension","SystolicBP","DiastolicBP","CholesterolTotal",
    "CholesterolLDL","CholesterolHDL","CholesterolTriglycerides","MMSE","FunctionalAssessment",
    "MemoryComplaints","BehavioralProblems","ADL","Confusion","Disorientation","PersonalityChanges",
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
    "PhysicalActivity": physical_activity,
    "DietQuality": diet_quality,
    "SleepQuality": sleep_quality,
    "FamilyHistoryAlzheimers": family_hist,
    "CardiovascularDisease": cardio,
    "Diabetes": diabetes,
    "Depression": depression,
    "HeadInjury": head_injury,
    "Hypertension": hypertension,
    "SystolicBP": systolic,
    "DiastolicBP": diastolic,
    "CholesterolTotal": chol_total,
    "CholesterolLDL": chol_ldl,
    "CholesterolHDL": chol_hdl,
    "CholesterolTriglycerides": triglycerides,
    "MMSE": mmse,
    "FunctionalAssessment": functional,
    "MemoryComplaints": memory_complaints,
    "BehavioralProblems": behavioral,
    "ADL": adl,
    "Confusion": confusion,
    "Disorientation": disorientation,
    "PersonalityChanges": personality_changes,
    "DifficultyCompletingTasks": difficulty_tasks,
    "Forgetfulness": forgetfulness
}

numeric_cols = [
    "Age","BMI","AlcoholConsumption","PhysicalActivity","DietQuality","SleepQuality",
    "SystolicBP","DiastolicBP","CholesterolTotal","CholesterolLDL","CholesterolHDL",
    "CholesterolTriglycerides","MMSE","FunctionalAssessment","ADL"
]

if st.button("Predict"):
    with st.spinner("Predicting..."):
        pred, proba = predict_and_proba(model, scaler, input_dict, numeric_cols, features_order)
        label, color = risk_label(proba)

        # --- SAFELY format label for display (no escaping issues) ---
        display_label = "Alzheimer's" if pred == 1 else "No Alzheimer"
        st.markdown(f"### Prediction: **{display_label}**")

        if proba is not None:
            st.markdown(f"**Risk probability:** {proba:.3f}")

        # colored risk strip
        st.markdown(
            f"<div style='padding:10px;background:{color};color:white;border-radius:6px;display:inline-block'>{label}</div>",
            unsafe_allow_html=True
        )

        st.subheader("Input Summary")
        st.dataframe(pd.DataFrame([input_dict]))

        if hasattr(model, "feature_importances_"):
            # try to map feature importances to features_order length
            try:
                imp = pd.Series(model.feature_importances_, index=features_order).sort_values(ascending=False)[:10]
                st.subheader("Top features (model importance)")
                st.bar_chart(imp)
            except Exception:
                # fallback if lengths mismatch
                st.info("Model has feature_importances_ but mapping failed (length mismatch).")
else:
    st.info("Fill patient features in the sidebar and click Predict.")

st.markdown("---")
st.caption("This app uses the trained Gradient Boosting model and StandardScaler.")
st.markdown("Built by Pushpam K. Kumari")

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")
st.caption("👩‍💻 Built by Pushpam Kumari | Model: Gradient Boosting | Deployed on Streamlit Cloud 🌐")
