import streamlit as st
import pandas as pd
import os
import sys
import gdown
import torch
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
from dataEngineer.modeling.Deeplearning2 import MultiOutputClassificationModel


st.set_page_config(page_title="Prediction Page", layout="wide",page_icon="📈")

left, center, right = st.columns([1,4,1])
with center:
    st.title("Citizen Issue Prediction Page")
    st.write("")


if st.session_state.get('Citizen'):
    name = st.session_state['Citizen'][0]
    email = st.session_state['Citizen'][1]
    phone = st.session_state['Citizen'][2]
    comment = st.session_state['Citizen'][3]


    with st.popover("Citizen Details",width="stretch"):
        col1, col2 = st.columns([1,3])
        with col1:
            st.write(f"**Name:** {name}")
            st.write(f"**Email:** {email}")
        with col2:
            st.write(f"**Phone:** {phone}")
        st.write(f"**Comment:** {comment}")

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
    # DEEP LEARNING MODEL PREDICTION
    # ============================================================
    st.subheader("Deep Learning (BERT) Prediction")

    # ---- MODEL PATH ----
    DL_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/best_cv_classifier.pth")

    # ---- GOOGLE DRIVE DIRECT FILE LINK (FIX THIS) ----
    model_id = '1BHpawVMowc8D8yJAeFde6FS6Zq62OE07'
    DL_MODEL_URL = f"https://drive.google.com/uc?id={model_id}"   # <-- replace with real ID

    # ---- DOWNLOAD MODEL IF MISSING ----
    if not os.path.exists(DL_MODEL_PATH):
        st.info("Downloading BERT Deep Learning model... this may take 10–20 seconds.")
        gdown.download(DL_MODEL_URL, DL_MODEL_PATH, quiet=False)

    # ---- LOAD MODEL (cached so it loads only once) ----
    @st.cache_resource
    def load_dl_model():
        return MultiOutputClassificationModel(
            model_name='distilbert-base-uncased',
            model_path=DL_MODEL_PATH
        )

    dl_model = load_dl_model()

    # ---- RUN PREDICTION ----
    try:
        dl_pred = dl_model.predict(comment)

        st.write("### Deep Learning Output")
        st.write(f"**Category:** {dl_pred['category']['prediction']} (Conf: {dl_pred['category']['confidence']:.2f})")
        st.write(f"**Sub-Category:** {dl_pred['sub_category']['prediction']} (Conf: {dl_pred['sub_category']['confidence']:.2f})")

        st.write("**Top Category Predictions:**")
        st.write(dl_pred['category']['top_predictions'])

        st.write("**Top Sub-Category Predictions:**")
        st.write(dl_pred['sub_category']['top_predictions'])

    except Exception as e:
        st.error("⚠️ Deep Learning model failed to run.")
        st.error(str(e))

    Category_confidence = dl_pred['category']['confidence']
    SubCategory_confidence = dl_pred['sub_category']['confidence']
    Average_confidence = (Category_confidence + SubCategory_confidence) / 2
    Ratio = (Category_confidence - SubCategory_confidence) / Category_confidence

    if st.session_state.get('Prediction'):
        st.markdown(f"## Prediction Results")
        col1, col2, col3 = st.columns([2,3,1])
        with col1:
            st.markdown(f"#### Category:",)
            st.markdown(f"#### Sub-Category:")
        with col2:
            st.markdown(f"#### {dl_pred['category']['prediction']}")
            st.markdown(f"#### {dl_pred['sub_category']['prediction']}")
        with col3:

            st.metric(label="Confidence", value=f"{Category_confidence * 100:.2f}%",delta= (f"{Ratio * 100:.2f}"))
    else:
        st.markdown(st.session_state.get('Prediction', "No prediction available. Please submit the form first."))



    # ============================================================
    # ELT NEW DATA
    # ============================================================


    new_data = {
            "category": dl_pred['category']['prediction'],
            "subreddit": dl_pred['sub_category']['prediction'],
            "problem_type": predictions['problem_type'][0],
            "title": comment,
            "text": comment
        }


    new_data_path = os.path.join(
        os.path.dirname(__file__),
        "../../data/new/new_data.csv"
    )

    if os.path.exists(new_data_path) and os.path.getsize(new_data_path) > 0:
        df = pd.read_csv(new_data_path)
    else:
        df = pd.DataFrame(columns=["category", "subreddit", "problem_type", "title", "text"])
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_csv(new_data_path, index=False)

else:
    st.warning("Please fill out the citizen form on the main page before making a prediction.")   