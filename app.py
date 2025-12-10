import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Page configuration
# ---------------------------------------------------
st.set_page_config(page_title="Solar Power Generation Predictor", page_icon="🔆", layout="centered")

# ---------------------------------------------------
# Helper Function (for color logic)
# ---------------------------------------------------
def power_level_and_color(pred: float):
    if pred < 2000:
        return "Low", "#ff6b6b"       # red
    elif pred < 4000:
        return "Moderate", "#f4c542"  # yellow
    else:
        return "High", "#4cd137"      # green

# ---------------------------------------------------
# Title & Intro
# ---------------------------------------------------
st.title("🔆 Solar Power Generation Predictor")
st.write("Enter the weather parameters below and get predicted **power-generated.**")

# ---------------------------------------------------
# Load Model and Scaler
# ---------------------------------------------------
@st.cache_resource
def load_artifacts():
    model_path = "best_model.pkl"
    scaler_path = "scaler.pkl"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("❌ Model or scaler file missing. Please upload `best_model.pkl` and `scaler.pkl`.")
        st.stop()
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts()

# ---------------------------------------------------
# Features
# ---------------------------------------------------
FEATURES = [
    "distance-to-solar-noon",
    "temperature",
    "wind-direction",
    "wind-speed",
    "sky-cover",
    "visibility",
    "humidity",
    "average-wind-speed-(period)",
    "average-pressure-(period)"
]

# ---------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------
st.sidebar.header("Input Parameters")

def sidebar_inputs():
    vals = {
        "distance-to-solar-noon": st.sidebar.number_input("distance-to-solar-noon (0–1)", 0.0, 1.0, 0.50, 0.01),
        "temperature": st.sidebar.number_input("temperature (°F)", -50, 150, 70, 1),
        "wind-direction": st.sidebar.number_input("wind-direction (deg)", 0, 360, 90, 1),
        "wind-speed": st.sidebar.number_input("wind-speed (mph)", 0.0, 100.0, 5.0, 0.1),
        "sky-cover": st.sidebar.number_input("sky-cover (0–100 or categorical scale)", 0, 100, 20, 1),
        "visibility": st.sidebar.number_input("visibility (miles)", 0.0, 20.0, 10.0, 0.1),
        "humidity": st.sidebar.number_input("humidity (%)", 0, 100, 50, 1),
        "average-wind-speed-(period)": st.sidebar.number_input("average-wind-speed-(period)", 0.0, 100.0, 5.0, 0.1),
        "average-pressure-(period)": st.sidebar.number_input("average-pressure-(period)", 0.0, 40.0, 29.8, 0.1),
    }
    return vals

user_vals = sidebar_inputs()

# ---------------------------------------------------
# Predict / Reset Buttons
# ---------------------------------------------------
if "pred" not in st.session_state:
    st.session_state.pred = None

c1, c2 = st.columns(2)

with c1:
    if st.button("🔮 Predict", use_container_width=True):
        row = pd.DataFrame([[user_vals[f] for f in FEATURES]], columns=FEATURES)
        X_scaled = scaler.transform(row)
        pred = float(model.predict(X_scaled)[0])
        st.session_state.pred = pred
        st.success(f"Estimated Power Generated: **{pred:,.0f}** units")

with c2:
    if st.button("✨ Reset to Defaults", use_container_width=True):
        st.session_state.pred = None
        st.rerun()

# ---------------------------------------------------
# Notes Section
# ---------------------------------------------------
st.markdown("""
---
**Notes**
- This app uses the trained Gradient Boosting model and StandardScaler.
- Feature order must match the training sequence.
""")

# ---------------------------------------------------
# About & Input Help
# ---------------------------------------------------
with st.expander("ℹ️ About this app"):
    st.markdown("""
- **Goal:** Forecast solar power from weather parameters using a trained regression model.  
- **Model Used:** Gradient Boosting (R² ≈ 0.90).  
- **Preprocessing:** Standardization via `StandardScaler`.  
- **Deployment:** Streamlit Cloud.  
- **How to use:** Adjust inputs on the left → click **Predict** → view power output.
""")

with st.expander("❓ What do the inputs mean?"):
    st.markdown("""
- **distance-to-solar-noon (0–1):** 0 = near noon (more sunlight).  
- **temperature (°F):** Higher temperature → more power.  
- **wind-speed (mph):** Moderate wind helps cool panels.  
- **sky-cover (0–100):** More clouds → less sunlight.  
- **visibility (miles):** Clearer sky → higher solar intensity.  
- **humidity (%):** High humidity absorbs light → reduces power output.  
""")

# ---------------------------------------------------
# Visualization (Bar + Trend + Legend)
# ---------------------------------------------------
if st.session_state.pred is not None:
    pred = st.session_state.pred
    level, color = power_level_and_color(pred)

    st.subheader("🌞 Power Generation Visualization")

    # Bar Chart
    fig, ax = plt.subplots(figsize=(6, 1.8))
    ax.barh(["Predicted Power"], [pred], color=color)
    ax.set_xlabel("Power (Units)")
    ax.set_xlim(0, max(pred * 1.5, 5000))
    try:
        ax.bar_label(ax.containers[0], fmt='%d', label_type='center', color="black", fontsize=10)
    except Exception:
        pass
    st.pyplot(fig)

    # Text based on Power Level
    st.markdown(f"### ⚡ Power Level: **{level}**")
    if level == "Low":
        st.warning("🌥️ Prediction indicates **low power generation** under current weather conditions.")
    elif level == "Moderate":
        st.info("🌤️ Prediction indicates **moderate power generation** — fair sunlight conditions.")
    else:
        st.success("☀️ Prediction indicates **high power generation** — ideal conditions for solar output!")
        st.balloons()

    # Legend Strip
    st.markdown("""
    <div style='display:flex; justify-content:space-evenly; text-align:center; margin-top:15px;'>
        <div style='background-color:#ff6b6b; width:60px; height:15px; border-radius:5px;'></div>
        <div style='background-color:#f4c542; width:60px; height:15px; border-radius:5px;'></div>
        <div style='background-color:#4cd137; width:60px; height:15px; border-radius:5px;'></div>
    </div>
    <div style='display:flex; justify-content:space-evenly; font-size:13px; margin-top:5px;'>
        <span>Low</span>
        <span>Moderate</span>
        <span>High</span>
    </div>
    """, unsafe_allow_html=True)

    # Mini Trend Chart
    st.markdown("##### 📈 Power Comparison Trend")
    trend_values = np.array([pred * 0.8, pred * 0.9, pred])
    st.line_chart(trend_values)

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")
st.caption("👩‍💻 Built by Pushpam Kumari | Model: Gradient Boosting | Deployed on Streamlit Cloud 🌐")
