import streamlit as st
import os
import sys

# ============================================================
# FIX PYTHON PATH SO IMPORTS WORK FROM gui/main/
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
sys.path.append(PROJECT_ROOT)

# ============================================================
# IMPORT YOUR PROJECT MODULES
# ============================================================
from dataEngineer.modeling.MLmodel2 import MultiTaskTextClassifier
from dataEngineer.pipeLine import *

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Citizens Issues Submission",
    layout="wide",
    page_icon="📝"
)

# ============================================================
# BACKGROUND CSS
# ============================================================
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://www.tripsavvy.com/thmb/-tzBp8Gy4A2v7-XK07TYecZXWfk=/2286x1311/filters:fill(auto,1)/GettyImages-200478089-001-06db86e7b540494a807a46af6c6c7f11.jpg");
    background-size: cover;
    animation: moveBackground 120s linear infinite alternate;
}

@keyframes moveBackground {
    0% { background-position: 0 0; }
    100% { background-position: 100% 0; }
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ============================================================
# PAGE TITLE
# ============================================================
st.title("Citizen Issues Submission")

# ============================================================
# FORM
# ============================================================
with st.form("client_form"):
    name = st.text_input("Client Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone Number")
    comment = st.text_area("Describe the problem in detail")

    submitted = st.form_submit_button("Submit")

# ============================================================
# HANDLE FORM SUBMISSION
# ============================================================
if submitted:

    # Store form data
    st.session_state['Citizen'] = [name, email, phone, comment]
    st.toast("Form submitted successfully!", icon="✅")

    # -------------------------
    # MODEL PATH (FIXED FOR DEPLOYMENT)
    # -------------------------
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models/my_multi_task_models_afterCleaning_logostic")

    tasks = ['problem_type', 'category']

    try:
        # Load model
        model = MultiTaskTextClassifier(
            label_columns=tasks,
            model_dir=MODEL_DIR,
            model_type='logreg',
            use_hyperparameter_tuning=True
        )

        # Predict
        new_texts = [comment]
        predictions = model.predict(new_texts)

        # Display results
        st.subheader("Predicted Classification")
        st.write(f"**Problem Type:** {predictions['problem_type'][0]}")
        st.write(f"**Category:** {predictions['category'][0]}")

        # Store prediction for another page if needed
        st.session_state['Prediction'] = [
            predictions['problem_type'][0],
            predictions['category'][0]
        ]

    except Exception as e:
        st.error("⚠️ Model failed to load. Make sure the 'models/' folder exists in your repo.")
        st.error(str(e))

# ============================================================
# FEEDBACK SECTION
# ============================================================
st.markdown("---")
col1, col2, col3 = st.columns([2, 2, 1])

with col2:
    st.markdown("###### Was this submission helpful?")

col1, col2, col3, g, t = st.columns([2, 3, 1, 3, 2])
with col3:
    selected = st.feedback("thumbs")
