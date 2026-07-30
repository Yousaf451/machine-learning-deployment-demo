# ============================================================
# FIFA World Cup 2026 Match Result Prediction
# Streamlit Web Application
# Part 1
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS
# ============================================================

st.markdown("""
<style>

.main{
    padding-top:20px;
}

h1{
    color:#0E4C92;
    text-align:center;
}

h2{
    color:#0E4C92;
}

.stButton>button{
    width:100%;
    background:#0E4C92;
    color:white;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.stDownloadButton>button{
    width:100%;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# Title
# ============================================================

st.title("⚽ FIFA World Cup 2026")

st.subheader("Player Match Result Prediction")

st.write(
"""
Predict the **Match Result**
using the trained Machine Learning model.

Supported Classes:

- Win
- Draw
- Loss
"""
)

# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("📌 Project Information")

st.sidebar.markdown("---")

st.sidebar.write("### Model")

st.sidebar.success("Logistic Regression / Random Forest")

st.sidebar.write("### Target Column")

st.sidebar.info("match_result")

st.sidebar.write("### Developed By")

st.sidebar.success("Muhammad Yousaf")

st.sidebar.markdown("---")

st.sidebar.write(
"""
Upload a CSV file
having the same columns
used during model training.
"""
)

# ============================================================
# Load Model
# ============================================================

from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_model" / "best_model.pkl"

# Debug (Temporary)
st.write("BASE_DIR:", BASE_DIR)
st.write("MODEL PATH:", MODEL_PATH)
st.write("Model Exists:", MODEL_PATH.exists())

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    st.sidebar.success("✅ Model Loaded Successfully")

except Exception as e:
    st.error("Unable to load model.")
    st.exception(e)
    st.stop()

# ============================================================
# Upload CSV
# ============================================================

uploaded_file = st.file_uploader(

    "📂 Upload CSV File",

    type=["csv"]

)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Uploaded Successfully")

    st.write("### Dataset Preview")

    st.dataframe(df.head())

    st.write("Rows :", df.shape[0])

    st.write("Columns :", df.shape[1])

    st.markdown("---")

    predict_button = st.button("🚀 Predict Match Result")
    # ============================================================
# Prediction
# ============================================================

    if predict_button:

        try:

            prediction = model.predict(df)

            result_df = df.copy()

            result_df["Predicted Match Result"] = prediction

            st.success("Prediction Completed Successfully!")

            st.markdown("## Prediction Results")

            st.dataframe(result_df)

            # ====================================================
            # Prediction Distribution
            # ====================================================

            st.markdown("## Prediction Distribution")

            prediction_counts = (
                result_df["Predicted Match Result"]
                .value_counts()
            )

            st.bar_chart(prediction_counts)

            # ====================================================
            # Prediction Probabilities
            # ====================================================

            if hasattr(model, "predict_proba"):

                st.markdown("## Prediction Probabilities")

                probabilities = model.predict_proba(df)

                class_names = model.classes_

                probability_df = pd.DataFrame(
                    probabilities,
                    columns=class_names
                )

                st.dataframe(probability_df)

            # ====================================================
            # Download Predictions
            # ====================================================

            csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(

                label="📥 Download Prediction CSV",

                data=csv,

                file_name="predictions.csv",

                mime="text/csv"

            )

        except Exception as e:

            st.error("Prediction Failed!")

            st.exception(e)

# ============================================================
# Footer
# ============================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;'>

### ⚽ FIFA World Cup 2026 Match Result Prediction

Built with ❤️ using

Python • Scikit-learn • Streamlit

Developed by <b>Muhammad Yousaf</b>

</div>
""",
unsafe_allow_html=True
)