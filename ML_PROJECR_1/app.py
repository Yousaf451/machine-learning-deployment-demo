"""
Streamlit Web Application for Adult Income Classification
with enhanced UI (cards, hero image, modern styling).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Income Classifier",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# CUSTOM CSS – same card styles, but adapted for Streamlit
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Main container */
        .main {
            padding: 1rem 2rem;
        }
        /* Hero image container */
        .hero-container {
            border-radius: 1.5rem;
            overflow: hidden;
            margin-bottom: 2rem;
            box-shadow: 0 8px 30px rgba(0,20,40,0.15);
        }
        .hero-container img {
            width: 100%;
            height: 220px;
            object-fit: cover;
            display: block;
        }
        /* Cards */
        .card {
            background: #ffffff;
            padding: 1.8rem 2rem;
            border-radius: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.04);
            border: 1px solid #eef2f6;
            margin-bottom: 1.8rem;
        }
        .card-header {
            font-size: 1.15rem;
            font-weight: 600;
            color: #0b1a33;
            margin-bottom: 1.25rem;
            border-bottom: 2px solid #f1f4f9;
            padding-bottom: 0.7rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }
        .card-header i {
            color: #2563eb;
        }
        /* Prediction box */
        .prediction-box {
            padding: 1.8rem 2rem;
            border-radius: 1.25rem;
            text-align: center;
            font-size: 1.5rem;
            font-weight: 700;
            margin-top: 0.5rem;
        }
        .prediction-positive {
            background: linear-gradient(135deg, #d1fae5, #a7f3d0);
            color: #065f46;
            border: 1px solid #6ee7b7;
        }
        .prediction-negative {
            background: linear-gradient(135deg, #fee2e2, #fecaca);
            color: #991b1b;
            border: 1px solid #fca5a5;
        }
        .confidence-bar {
            height: 0.65rem;
            background: #e2e8f0;
            border-radius: 100px;
            overflow: hidden;
            margin-top: 0.3rem;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #2563eb, #7c3aed);
            border-radius: 100px;
            transition: width 0.8s ease;
        }
        .footer {
            margin-top: 2.5rem;
            text-align: center;
            color: #94a3b8;
            border-top: 1px solid #eef2f6;
            padding-top: 1.8rem;
            font-size: 0.85rem;
        }
        .stButton button {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            font-weight: 600;
            border-radius: 100px;
            padding: 0.7rem 2rem;
            border: none;
            width: 100%;
            font-size: 1.1rem;
            transition: all 0.25s ease;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(37,99,235,0.35);
        }
        /* Form styling */
        .stSelectbox, .stNumberInput {
            margin-bottom: 0.5rem;
        }
        .st-emotion-cache-1y4p8pa {
            max-width: 100%;
        }
        /* Sidebar info */
        .sidebar-info {
            background: #f8fafc;
            padding: 1.2rem 1.5rem;
            border-radius: 1.25rem;
            border: 1px solid #e9edf2;
            margin-bottom: 1.5rem;
        }
        .sidebar-info p {
            margin: 0.3rem 0;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# LOAD MODEL & METADATA (cached)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model_and_metadata():
    BASE_DIR = Path(__file__).resolve().parent

    model_path = BASE_DIR / "best_classification_model.joblib"
    metadata_path = BASE_DIR / "model_metadata.joblib"

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()

    if not metadata_path.exists():
        st.error(f"Metadata file not found: {metadata_path}")
        st.stop()

    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)

    return model, metadata

    if not model_path.exists():
        st.error("Model file not found. Please ensure 'best_classification_model.joblib' is in the current directory.")
        st.stop()
    if not metadata_path.exists():
        st.error("Metadata file not found. Please ensure 'model_metadata.joblib' is in the current directory.")
        st.stop()

    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    return model, metadata

def extract_categories_from_model(model):
    preprocessor = model.named_steps['preprocessor']
    cat_transformer = preprocessor.named_transformers_['cat']
    onehot = cat_transformer.named_steps['onehot']
    return onehot.categories_

def get_classes_from_model(model):
    return model.named_steps['model'].classes_

# ----------------------------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------------------------
def main():
    model, metadata = load_model_and_metadata()

    feature_names = metadata['feature_columns']
    numeric_features = metadata['numeric_features']
    categorical_features = metadata['categorical_features']
    target_column = metadata['target_column']
    model_name = metadata.get('best_model_name', 'Unknown')

    categories_list = extract_categories_from_model(model)
    cat_categories = {
        cat_feat: list(cats) for cat_feat, cats in zip(categorical_features, categories_list)
    }

    target_classes = get_classes_from_model(model)
    positive_class = '>50K' if '>50K' in target_classes else target_classes[-1]

    # --------------------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## 📋 Instructions")
        st.markdown(
            """
            Fill in the details below and click **Predict Income**.
            The model predicts whether income exceeds $50K/yr.
            """
        )
        st.markdown("---")
        st.markdown("### Model Information")
        st.markdown(f"- **Model:** `{model_name}`")
        st.markdown(f"- **Features:** {len(feature_names)}")
        st.markdown(f"- **Target classes:** {', '.join(target_classes)}")
        if 'training_date' in metadata:
            st.markdown(f"- **Training date:** {metadata['training_date']}")
        st.markdown("---")
        st.caption("Built with Streamlit & scikit-learn")

    # --------------------------------------------------------------------
    # HERO IMAGE
    # --------------------------------------------------------------------
    # Use the same image URL (or you can store it locally)
    hero_url = "https://raw.githubusercontent.com/Yousaf451/machine-learning-deployment-demo/main/ML_PROJECR_1/ChatGPT%20Image%20Jul%2028%2C%202026%2C%2010_22_57%20PM.png"
    st.markdown(
        f"""
        <div class="hero-container">
            <img src="{hero_url}" alt="Income Classification Banner" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------
    # MAIN CONTENT
    # --------------------------------------------------------------------
    st.markdown("## 💰 Income Classification")
    st.markdown("Enter the individual's attributes below.")

    # Input card
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-header"><i class="fas fa-address-card"></i> Personal & Employment Details</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        input_values = {}

        with col1:
            st.markdown("#### 📊 Numerical Attributes")
            for feat in numeric_features:
                # For simplicity, use integer step for all numeric features
                value = st.number_input(
                    f"{feat}",
                    min_value=0,
                    value=0,
                    step=1,
                    key=f"num_{feat}",
                )
                input_values[feat] = value

        with col2:
            st.markdown("#### 🏷️ Categorical Attributes")
            for feat in categorical_features:
                options = cat_categories[feat]
                value = st.selectbox(
                    f"{feat}",
                    options=options,
                    index=0,
                    key=f"cat_{feat}",
                )
                input_values[feat] = value

        st.markdown('</div>', unsafe_allow_html=True)

    # Predict button
    predict_clicked = st.button("🔮 Predict Income", type="primary", use_container_width=True)

    # --------------------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------------------
    if predict_clicked:
        try:
            input_df = pd.DataFrame([input_values])
            input_df = input_df[feature_names]

            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]

            class_to_idx = {cls: idx for idx, cls in enumerate(target_classes)}
            pos_idx = class_to_idx.get(positive_class, 1)
            confidence = prediction_proba[pos_idx] * 100

            is_positive = (prediction == positive_class)

            # Display result in a card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="card-header"><i class="fas fa-chart-simple"></i> Prediction Result</div>',
                unsafe_allow_html=True,
            )

            if is_positive:
                box_class = "prediction-positive"
                label_text = f"✅ Predicted Income: **{prediction}** (Above $50K)"
            else:
                box_class = "prediction-negative"
                label_text = f"❌ Predicted Income: **{prediction}** (At or below $50K)"

            st.markdown(
                f'<div class="prediction-box {box_class}">{label_text}</div>',
                unsafe_allow_html=True,
            )

            # Confidence bar
            st.markdown(f"**Confidence:** {confidence:.2f}%")
            st.markdown(
                f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence:.2f}%;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Expandable details
            with st.expander("📈 Prediction Details"):
                prob_df = pd.DataFrame({
                    "Class": target_classes,
                    "Probability": prediction_proba * 100
                })
                st.dataframe(prob_df.style.format({"Probability": "{:.2f}"}))
                st.markdown(f"**Raw Prediction:** {prediction}")
                st.markdown(f"**Positive class:** {positive_class}")

            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.stop()

    # --------------------------------------------------------------------
    # EXPANDERS (Input Summary & Model Info)
    # --------------------------------------------------------------------
    with st.expander("📋 Input Summary"):
        if input_values:
            summary_df = pd.DataFrame({
                "Feature": list(input_values.keys()),
                "Value": list(input_values.values())
            })
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("No inputs entered yet.")

    with st.expander("🤖 Model Information"):
        st.markdown(f"**Model Type:** {model_name}")
        st.markdown(f"**Number of Features:** {len(feature_names)}")
        st.markdown(f"**Target Classes:** {', '.join(target_classes)}")
        st.markdown(f"**Target Column:** {target_column}")
        if 'training_date' in metadata:
            st.markdown(f"**Training Date:** {metadata['training_date']}")
        st.markdown("**Feature List:**")
        st.write(feature_names)

    # --------------------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------------------
    st.markdown(
        """
        <div class="footer">
            Built with ❤️ using Streamlit • Adult Income Dataset • Model: Gradient Boosting
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()