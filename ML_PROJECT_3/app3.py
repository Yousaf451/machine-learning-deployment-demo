import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

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
# Custom CSS (Modern UI, Glassmorphism & Custom Badges)
# ============================================================
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Banner Styling */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0e4c92 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto 1.5rem auto;
    }

    /* Badge Pills for Classes */
    .badge-container {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-top: 10px;
    }
    
    .badge {
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .badge-win { background-color: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
    .badge-draw { background-color: #fef9c3; color: #a16207; border: 1px solid #fef08a; }
    .badge-loss { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }

    /* Sidebar Cards */
    .sidebar-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(229, 231, 235, 0.2);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    
    .sidebar-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 4px;
    }
    
    .sidebar-value {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1e293b;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #0e4c92 0%, #0284c7 100%);
        color: white;
        border: none;
        border-radius: 10px;
        height: 50px;
        font-size: 16px;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(14, 76, 146, 0.25);
        transition: all 0.2s ease-in-out;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(14, 76, 146, 0.35);
    }

    .stDownloadButton>button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar Configuration
# ============================================================
st.sidebar.markdown("""
    <div style="text-align: center; padding-bottom: 10px;">
        <h2 style="margin: 0; color: #0E4C92;">⚽ Predictor Dashboard</h2>
        <p style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">FIFA World Cup 2026 ML Engine</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Information Cards
st.sidebar.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-label">Model Architecture</div>
        <div class="sidebar-value">Logistic Regression / Random Forest</div>
    </div>
    <div class="sidebar-card">
        <div class="sidebar-label">Target Column</div>
        <div class="sidebar-value"><code>match_result</code></div>
    </div>
    <div class="sidebar-card">
        <div class="sidebar-label">Developer</div>
        <div class="sidebar-value">Muhammad Yousaf</div>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Instructions:** Upload a CSV file matching the schema used during training.")

# ============================================================
# Model Loading & Path Handlers
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_model" / "best_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Load the model silently
try:
    model = load_model()
    st.sidebar.success("✅ Model Ready")
except Exception as e:
    st.error("❌ Unable to load the trained Machine Learning model.")
    st.exception(e)
    st.stop()

# ============================================================
# Main Content: Hero Header
# ============================================================
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">⚽ Player Match Result Prediction</div>
        <div class="hero-subtitle">
            Upload match statistics to predict game outcomes using our trained ensemble machine learning pipeline.
        </div>
        <div class="badge-container">
            <span class="badge badge-win">🟢 Win</span>
            <span class="badge badge-draw">🟡 Draw</span>
            <span class="badge badge-loss">🔴 Loss</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ============================================================
# Advanced Settings Accordion (Hides Raw Paths)
# ============================================================
with st.expander("🛠️ Advanced Settings & System Diagnostics"):
    st.write("**System Diagnostics & Paths:**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Base Directory", str(BASE_DIR.name))
    col2.metric("Model Status", "Found" if MODEL_PATH.exists() else "Missing")
    col3.metric("Model Path", str(MODEL_PATH.name))

# ============================================================
# File Upload Section
# ============================================================
st.markdown("### 📂 Data Input")
uploaded_file = st.file_uploader(
    "Upload CSV dataset for batch prediction",
    type=["csv"],
    help="Make sure the CSV matches the exact features format required by the model."
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.success("✅ Dataset Uploaded Successfully")
    
    # Dataset Metadata Cards
    m1, m2 = st.columns(2)
    m1.metric(label="Total Rows", value=df.shape[0])
    m2.metric(label="Total Columns", value=df.shape[1])
    
    with st.expander("👀 Preview Input Dataset", expanded=True):
        st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")
    predict_button = st.button("🚀 Predict Match Results")

    # ============================================================
    # Prediction Logic & Output
    # ============================================================
    if predict_button:
        try:
            with st.spinner("Running predictions..."):
                prediction = model.predict(df)
                result_df = df.copy()
                result_df["Predicted Match Result"] = prediction

            st.balloons()
            st.success("✨ Prediction Completed Successfully!")

            # Prediction Results Section
            st.markdown("## 📊 Prediction Results")
            st.dataframe(result_df, use_container_width=True)

            # Visualizations Layout
            col_chart, col_prob = st.columns([1, 1])

            with col_chart:
                st.markdown("### 📈 Prediction Distribution")
                prediction_counts = result_df["Predicted Match Result"].value_counts()
                st.bar_chart(prediction_counts)

            with col_prob:
                if hasattr(model, "predict_proba"):
                    st.markdown("### 🎯 Probabilities")
                    probabilities = model.predict_proba(df)
                    class_names = model.classes_
                    probability_df = pd.DataFrame(probabilities, columns=class_names)
                    st.dataframe(probability_df, use_container_width=True)

            # Download Section
            st.markdown("---")
            csv = result_df.to_csv(index=False).encode("utf-8")
            
            st.download_button(
                label="📥 Download Full Prediction CSV",
                data=csv,
                file_name="match_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("❌ Prediction Failed!")
            st.exception(e)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color: #64748b; font-size: 0.9rem; padding: 10px;'>
    ⚽ <b>FIFA World Cup 2026 Match Result Prediction</b><br>
    Built with ❤️ using Python • Scikit-learn • Streamlit<br>
    Developed by <b>Muhammad Yousaf</b>
</div>
""", unsafe_allow_html=True)

#live link: https://machine-learning-deployment-demo-4zjm9d2anoxchhgkwhhwyz.streamlit.app/