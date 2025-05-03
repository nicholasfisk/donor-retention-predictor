"""
app.py
------
Streamlit app that loads the trained donor-retention model
(models/model.pkl) and lets a user score one donor at a time.
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import shap_utils  

MODEL_PATH = Path("models/model.pkl")

@st.cache_resource  # reload only if the file changes
def load_model():
    return joblib.load(MODEL_PATH)

pipe = load_model()
st.title("🩸 Donor-Retention Predictor")

st.markdown("Fill in the donor’s details and click **Predict**")

# ---------- Input widgets -------------------------------------------------
recency_months   = st.number_input("Months since last donation",      0, 120, 2)
frequency        = st.number_input("Total number of donations",       1, 50, 4)
monetary_cc      = st.number_input("Total blood donated (cc)",        250, 2500, 750, step=50)
time_months      = st.number_input("Months since first donation",     1, 300, 50)

email_open_rate  = st.slider("Email open-rate", 0.0, 1.0, 0.35, 0.01)
events_attended  = st.slider("Events attended", 0, 10, 1)
sms_opt_in       = st.selectbox("SMS opt-in?", ["No", "Yes"]) == "Yes"
last_campaign    = st.number_input("Days since last campaign", 0, 365, 30)

segment          = st.selectbox("Newsletter segment",
                                ["Bronze", "Silver", "Gold", "Platinum"])

inputs = pd.DataFrame([{
    "recency_months":   recency_months,
    "frequency":        frequency,
    "monetary_cc":      monetary_cc,
    "time_months":      time_months,
    "email_open_rate":  email_open_rate,
    "events_attended":  events_attended,
    "sms_opt_in":       int(sms_opt_in),
    "last_campaign_days": last_campaign,
    "newsletter_segment": segment,
}])

if st.button("Predict"):
    proba = pipe.predict_proba(inputs)[0, 1]
    st.metric("Probability donor gives again", f"{proba*100:.1f}%")
    st.progress(proba)
    # --- SHAP explanation ------------------------------------------------
    with st.expander("Explain this prediction", expanded=False):
        explainer, feat_names = shap_utils.shap_explainer(pipe)

        # Pre-process the single input row to match training feature space
        X_input_proc, _ = shap_utils._preprocess_with_feature_names(
            pipe, inputs
        )
        shap_values = explainer(X_input_proc)

        fig = plt.figure()
        shap.plots.waterfall(
            shap_values[0],
            max_display=10,      # show all 10 original + any one-hot cols
            show=False,
        )
        st.pyplot(fig)
        plt.close(fig)