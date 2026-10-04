import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Load trained model and scaler
# -----------------------------

scaler = joblib.load("scaler.pkl")
model = joblib.load("isolation_forest.pkl")


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Industrial Compressor Monitoring",
    page_icon="⚙️",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("⚙️ Industrial Compressor Condition Monitoring")

st.write(
    "Detect abnormal compressor operating conditions "
    "using an Isolation Forest machine learning model."
)


# -----------------------------
# Sensor inputs
# -----------------------------

st.subheader("Enter Sensor Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    TP2 = st.number_input("TP2", value=1.2)
    TP3 = st.number_input("TP3", value=1.5)
    H1 = st.number_input("H1", value=1.1)
    DV_pressure = st.number_input("DV_pressure", value=0.0)
    Reservoirs = st.number_input("Reservoirs", value=1.4)

with col2:
    Oil_temperature = st.number_input("Oil_temperature", value=60.0)
    Motor_current = st.number_input("Motor_current", value=2.1)
    COMP = st.number_input("COMP", value=1.0)
    DV_eletric = st.number_input("DV_eletric", value=0.0)
    Towers = st.number_input("Towers", value=1.0)

with col3:
    MPG = st.number_input("MPG", value=1.0)
    LPS = st.number_input("LPS", value=1.0)
    Pressure_switch = st.number_input("Pressure_switch", value=1.0)
    Oil_level = st.number_input("Oil_level", value=1.0)
    Caudal_impulses = st.number_input("Caudal_impulses", value=1.0)


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔍 Detect Condition", use_container_width=True):

    sensor_data = {
        "TP2": TP2,
        "TP3": TP3,
        "H1": H1,
        "DV_pressure": DV_pressure,
        "Reservoirs": Reservoirs,
        "Oil_temperature": Oil_temperature,
        "Motor_current": Motor_current,
        "COMP": COMP,
        "DV_eletric": DV_eletric,
        "Towers": Towers,
        "MPG": MPG,
        "LPS": LPS,
        "Pressure_switch": Pressure_switch,
        "Oil_level": Oil_level,
        "Caudal_impulses": Caudal_impulses
    }

    # Convert input into DataFrame
    input_data = pd.DataFrame([sensor_data])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Get anomaly score
    anomaly_score = model.decision_function(input_scaled)[0]


    # -----------------------------
    # Display result
    # -----------------------------

    st.subheader("Prediction Result")

    if prediction == -1:

        st.error("🚨 ANOMALY DETECTED")

        st.write(
            "The compressor operating condition appears "
            "different from the learned normal behavior."
        )

    else:

        st.success("✅ NORMAL CONDITION")

        st.write(
            "The compressor operating condition appears "
            "consistent with the learned normal behavior."
        )


    st.metric(
        "Anomaly Score",
        round(float(anomaly_score), 4)
    )
