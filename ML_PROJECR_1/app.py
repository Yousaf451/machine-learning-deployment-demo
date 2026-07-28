```python
"""
Streamlit Web Application for Adult Income Classification
Uses the pre-trained model and metadata without retraining.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Income Classifier",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# 2. CUSTOM CSS
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* main container */
        .main {
            padding: 1rem 2rem;
        }
        /* card styling */
        .card {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            margin-bottom: 1.5rem;
            border: 1px solid #e9ecef;
        }
        .card-header {
            font-size: 1.25rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 0.75rem;
            border-bottom: 1px solid #e9ecef;
            padding-bottom: 0.5rem;
        }
        .prediction-box {
            padding: 1.5rem;
            border-radius: 0.75rem;
            text-align: center;
            font-size: 1.5rem;
            font-weight: 600;
            margin-top: 1rem;
            background-color: #f8f9fa;
        }
        .prediction-positive {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .prediction-negative {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .stButton button {
            background-color: #007bff;
            color: white;
            font-weight: 500;
            border-radius: 0.5rem;
            padding: 0.5rem 2rem;
            border: none;
            transition: background-color 0.2s;
        }
        .stButton button:hover {
            background-color: #0069d9;
            color: white;
        }
        .footer {
            margin-top: 3rem;
            text-align: center;
            color: #6c757d;
            font-size: 0.9rem;
            border-top: 1px solid #dee2e6;
            padding-top: 1.5rem;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        /* style select boxes and number inputs */
        .stSelectbox, .stNumberInput {
            margin-bottom: 0.75rem;
        }
        .st-emotion-cache-1y4p8pa {
            max-width: 100%;
        }
        .st-emotion-cache-1v0mbdj {
            background-color: #f8f9fa;
        }
        hr {
            margin: 1rem 0;
        }
        .label-highlight {
            font-weight: 500;
            color: #1f2937;
        }
        .confidence-bar {
            margin-top: 0.5rem;
            height: 1rem;
            background-color: #e9ecef;
            border-radius: 0.5rem;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background-color: #007bff;
            border-radius: 0.5rem;
            transition: width 0.3s;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# 3. LOAD RESOURCES (cached)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model_and_metadata():
    """Load the saved model and metadata from joblib files."""
    model_path = Path("best_classification_model.joblib")
    metadata_path = Path("model_metadata.joblib")

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
    """
    Extract the categories for each categorical feature from the fitted
    OneHotEncoder inside the pipeline.
    Returns a dict: feature_name -> list of categories.
    """
    # The pipeline: preprocessor -> model
    preprocessor = model.named_steps['preprocessor']
    # Get the categorical transformer
    cat_transformer = preprocessor.named_transformers_['cat']
    # The one-hot encoder
    onehot = cat_transformer.named_steps['onehot']
    # categories_ is a list of arrays, one per categorical feature
    # in the order of categorical_features from metadata
    categories_list = onehot.categories_
    return categories_list


def get_classes_from_model(model):
    """Return the target class labels from the model."""
    # The final estimator is a classifier with classes_ attribute
    estimator = model.named_steps['model']
    return estimator.classes_


# ----------------------------------------------------------------------------
# 4. MAIN APP
# ----------------------------------------------------------------------------
def main():
    # Load model and metadata
    model, metadata = load_model_and_metadata()

    # Extract information from metadata
    feature_names = metadata['feature_columns']      # list of all feature names in order
    numeric_features = metadata['numeric_features']  # list of numeric columns
    categorical_features = metadata['categorical_features']  # list of categorical columns
    target_column = metadata['target_column']
    model_name = metadata.get('best_model_name', 'Unknown')

    # Extract categories from the fitted encoder
    categories_list = extract_categories_from_model(model)
    # Build a dict mapping categorical feature -> list of categories
    cat_categories = {
        cat_feat: list(cats) for cat_feat, cats in zip(categorical_features, categories_list)
    }

    # Get target classes
    target_classes = get_classes_from_model(model)  # e.g., ['<=50K', '>50K']
    # Determine which class is the positive one (assuming '>50K' is positive)
    # We'll use the second class if it's >50K, otherwise we'll assume the positive is the one with higher income
    # For safety, we'll check if '>50K' in target_classes
    positive_class = None
    if '>50K' in target_classes:
        positive_class = '>50K'
    elif len(target_classes) == 2:
        # Usually the second is positive if sorted alphabetically: ['<=50K', '>50K']
        positive_class = target_classes[1]
    else:
        positive_class = target_classes[-1]  # fallback

    # ------------------------------------------------------------------------
    # SIDEBAR
    # ------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## 📋 Instructions")
        st.markdown(
            """
            Fill in the details of the individual below and click
            **Predict Income** to get a classification.

            The model was trained on the **Adult Income** dataset
            and predicts whether income exceeds $50K/yr.
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

    # ------------------------------------------------------------------------
    # MAIN AREA
    # ------------------------------------------------------------------------
    st.markdown("## 💰 Income Classification")
    st.markdown("Enter the individual's attributes below.")

    # Create a card for input fields
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">👤 Personal & Employment Details</div>', unsafe_allow_html=True)

        # We'll create two columns for better layout
        col1, col2 = st.columns(2)

        # We need to collect values in a dict keyed by feature name
        input_values = {}

        # Iterate over all features in the order they appear in feature_names
        # but we will place numeric and categorical in separate columns to balance layout
        # However, we must maintain the correct order for prediction.
        # We'll store inputs in a dict with original feature names.

        # First, collect all numeric inputs in col1 and categorical in col2?
        # But we need to maintain the order; we can just create a dictionary and later create a DataFrame in correct order.

        # Determine which features go to which column: we can alternate or put numeric in col1, categorical in col2.
        # For simplicity, we'll put numeric in col1 and categorical in col2.
        with col1:
            st.markdown("#### 📊 Numerical Attributes")
            for feat in numeric_features:
                # Determine if integer or float
                # We can't know dtype from metadata, but we can infer from feature name or use step=1 for integer-like
                # We'll treat all as float, but for age, education-num, hours-per-week, capital-gain, capital-loss, fnlwgt are ints
                # We'll use step=1 for those that are likely integers, else step=0.01
                # We'll just use step=1 for all numeric for simplicity because they are integers in this dataset.
                # But we can check if the feature name contains 'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week' - all ints.
                # So we'll set step=1 for all numeric.
                # However, we don't want to hardcode, but we can set step=1 for all.
                # Alternatively, we can just use number_input with default step=1.
                # To be safe, we'll use step=1 for all.
                value = st.number_input(
                    f"{feat}",
                    min_value=0,
                    value=0,
                    step=1,
                    key=f"num_{feat}",
                    help=f"Enter the value for {feat}"
                )
                input_values[feat] = value

        with col2:
            st.markdown("#### 🏷️ Categorical Attributes")
            for feat in categorical_features:
                options = cat_categories[feat]
                default_index = 0
                # Set a sensible default (e.g., the most frequent category)
                # We could also use the first category
                value = st.selectbox(
                    f"{feat}",
                    options=options,
                    index=default_index,
                    key=f"cat_{feat}",
                    help=f"Select a value for {feat}"
                )
                input_values[feat] = value

        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------------------------------------
    # PREDICT BUTTON
    # ------------------------------------------------------------------------
    predict_clicked = st.button("🔮 Predict Income", type="primary", use_container_width=True)

    # ------------------------------------------------------------------------
    # PREDICTION SECTION
    # ------------------------------------------------------------------------
    if predict_clicked:
        try:
            # 1. Create a DataFrame with one row, columns in the exact order of feature_names
            # We need to ensure we have all features
            input_df = pd.DataFrame([input_values])  # dict keys are feature names
            # Reorder columns to match feature_names
            input_df = input_df[feature_names]

            # 2. Predict
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]  # array of probabilities for each class

            # 3. Determine confidence for positive class
            # Map class to index
            class_to_idx = {cls: idx for idx, cls in enumerate(target_classes)}
            pos_idx = class_to_idx.get(positive_class, 1)  # fallback to index 1
            confidence = prediction_proba[pos_idx] * 100

            # 4. Display results in a card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">📊 Prediction Result</div>', unsafe_allow_html=True)

            # Determine color
            is_positive = (prediction == positive_class)
            if is_positive:
                box_class = "prediction-positive"
                label_text = f"✅ Predicted Income: **{prediction}** (Above $50K)"
            else:
                box_class = "prediction-negative"
                label_text = f"❌ Predicted Income: **{prediction}** (At or below $50K)"

            st.markdown(
                f'<div class="prediction-box {box_class}">{label_text}</div>',
                unsafe_allow_html=True
            )

            # Show confidence
            st.markdown(f"**Confidence:** {confidence:.2f}%")
            # Progress bar
            st.markdown(
                f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence:.2f}%;"></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Additional details expander
            with st.expander("📈 Prediction Details"):
                # Show probabilities for each class
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

    # ------------------------------------------------------------------------
    # EXPANDER: Input Summary
    # ------------------------------------------------------------------------
    with st.expander("📋 Input Summary"):
        if input_values:
            # Show the values in a nice table
            summary_df = pd.DataFrame({
                "Feature": list(input_values.keys()),
                "Value": list(input_values.values())
            })
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("No inputs entered yet.")

    # ------------------------------------------------------------------------
    # EXPANDER: Model Information
    # ------------------------------------------------------------------------
    with st.expander("🤖 Model Information"):
        st.markdown(f"**Model Type:** {model_name}")
        st.markdown(f"**Number of Features:** {len(feature_names)}")
        st.markdown(f"**Target Classes:** {', '.join(target_classes)}")
        st.markdown(f"**Target Column:** {target_column}")
        if 'training_date' in metadata:
            st.markdown(f"**Training Date:** {metadata['training_date']}")
        st.markdown("**Feature List:**")
        st.write(feature_names)

    # ------------------------------------------------------------------------
    # FOOTER
    # ------------------------------------------------------------------------
    st.markdown(
        """
        <div class="footer">
            Built with ❤️ using Streamlit • Adult Income Dataset • Model: Gradient Boosting
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# 5. ENTRY POINT
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
```