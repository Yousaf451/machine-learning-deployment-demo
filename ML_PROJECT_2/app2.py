# =============================================
# 1. IMPORTS & CONFIG
# =============================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# 2. LOAD ARTIFACTS (cached)
# =============================================
@st.cache_resource
def load_artifacts():
    model_path = "models/improved_nn_model.keras"
    preprocessor_path = "models/preprocessor.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        st.error("❌ Model or preprocessor not found. Please train the model first.")
        return None, None
    
    model = tf.keras.models.load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

model, preprocessor = load_artifacts()

# =============================================
# 3. HELPER FUNCTIONS
# =============================================
def predict_price_from_df(input_df, model, preprocessor):
    """Predict price from a DataFrame (raw features)."""
    try:
        X_processed = preprocessor.transform(input_df)
        X_processed = X_processed.toarray().astype(np.float32)
        log_pred = model.predict(X_processed).flatten()
        price_pred = np.expm1(log_pred)   # inverse log1p
        return price_pred
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# =============================================
# 4. SIDEBAR NAVIGATION
# =============================================
st.sidebar.title("🏠 Navigation")
option = st.sidebar.radio(
    "Choose an option:",
    ["Single Prediction", "Batch Prediction (CSV)", "Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Enter listing details in the form below and click **Predict** to get an instant price estimate."
)

# =============================================
# 5. SINGLE PREDICTION (FORM-BASED)
# =============================================
if option == "Single Prediction":
    st.title("🔍 NYC Airbnb Price Predictor")
    st.markdown("Fill in the listing details below and click **Predict** to get an estimated price.")
    
    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            neighbourhood_group = st.selectbox(
                "Neighbourhood Group",
                ["Brooklyn", "Manhattan", "Queens", "Staten Island", "Bronx"],
                help="Which borough is the listing in?"
            )
            neighbourhood = st.text_input(
                "Neighbourhood",
                value="Midtown",
                help="Specific neighbourhood name (e.g., Midtown, Williamsburg)"
            )
            latitude = st.number_input(
                "Latitude",
                min_value=40.0, max_value=41.0, value=40.75, format="%.5f",
                help="Geographic latitude (approx 40.5–40.9 for NYC)"
            )
            longitude = st.number_input(
                "Longitude",
                min_value=-74.3, max_value=-73.7, value=-73.98, format="%.5f",
                help="Geographic longitude (approx -74.0 – -73.8 for NYC)"
            )
            room_type = st.selectbox(
                "Room Type",
                ["Entire home/apt", "Private room", "Shared room"]
            )
        
        with col2:
            minimum_nights = st.number_input(
                "Minimum Nights",
                min_value=1, value=1, step=1,
                help="Minimum number of nights required for booking"
            )
            number_of_reviews = st.number_input(
                "Number of Reviews",
                min_value=0, value=0, step=1,
                help="Total reviews received so far"
            )
            reviews_per_month = st.number_input(
                "Reviews per Month",
                min_value=0.0, value=0.0, step=0.1,
                help="Average number of reviews per month"
            )
            calculated_host_listings_count = st.number_input(
                "Host Listings Count",
                min_value=1, value=1, step=1,
                help="Total listings the host has"
            )
            availability_365 = st.number_input(
                "Availability (days/year)",
                min_value=0, max_value=365, value=365, step=1,
                help="Number of days the listing is available in a year"
            )
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)
        
        if submitted:
            # Build a DataFrame with exactly one row
            input_data = {
                "neighbourhood_group": [neighbourhood_group],
                "neighbourhood": [neighbourhood],
                "latitude": [latitude],
                "longitude": [longitude],
                "room_type": [room_type],
                "minimum_nights": [minimum_nights],
                "number_of_reviews": [number_of_reviews],
                "reviews_per_month": [reviews_per_month],
                "calculated_host_listings_count": [calculated_host_listings_count],
                "availability_365": [availability_365]
            }
            input_df = pd.DataFrame(input_data)
            
            with st.spinner("Predicting..."):
                pred = predict_price_from_df(input_df, model, preprocessor)
                if pred is not None:
                    # Display result in a nice card
                    st.markdown("---")
                    st.markdown("### ✅ Prediction Result")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown(
                            f"""
                            <div style="background-color:#f0f8ff; padding:20px; border-radius:10px; text-align:center;">
                                <h2 style="color:#0066cc; margin:0;">${pred[0]:.2f}</h2>
                                <p style="color:#666; margin:0;">Estimated price per night</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    # Input summary - FIXED: removed .style.highlight_max(axis=0)
                    with st.expander("📋 Input Summary"):
                        st.dataframe(input_df.T)   # plain table, no styling errors
    # Example in sidebar
    with st.sidebar.expander("💡 Example"):
        st.markdown("Try these values for a Manhattan studio:")
        st.code("""
        Neighbourhood Group: Manhattan
        Neighbourhood: Midtown
        Latitude: 40.75362
        Longitude: -73.98377
        Room Type: Entire home/apt
        Min Nights: 3
        Reviews: 45
        Reviews/Month: 0.38
        Host Listings: 2
        Availability: 355
        """)

# =============================================
# 6. BATCH PREDICTION (CSV UPLOAD)
# =============================================
elif option == "Batch Prediction (CSV)":
    st.header("📊 Batch Prediction (CSV Upload)")
    st.markdown("Upload a CSV file with multiple listings. The file must contain the same columns as used in training.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", input_df.head())
        if st.button("Predict All"):
            with st.spinner("Processing..."):
                preds = predict_price_from_df(input_df, model, preprocessor)
                if preds is not None:
                    result_df = input_df.copy()
                    result_df['Predicted_Price'] = preds
                    st.success("Predictions completed!")
                    st.dataframe(result_df.style.highlight_max(axis=0))
                    
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions (CSV)",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                    fig, ax = plt.subplots(figsize=(8,4))
                    sns.histplot(result_df['Predicted_Price'], bins=30, kde=True, ax=ax)
                    ax.set_title("Distribution of Predicted Prices")
                    ax.set_xlabel("Price ($)")
                    st.pyplot(fig)

# =============================================
# 7. MODEL PERFORMANCE
# =============================================
else:
    st.header("📈 Model Performance")
    st.markdown("Evaluation metrics on the test set (held-out data).")
    
    try:
        df = pd.read_csv("AB_NYC_2019.csv")
        drop_cols = ['id', 'host_id', 'host_name', 'last_review', 'name']
        df = df.drop(columns=drop_cols, errors='ignore')
        X = df.drop(columns=['price'], errors='ignore')
        y = df['price']
        
        X_processed = preprocessor.transform(X).toarray().astype(np.float32)
        y_pred_log = model.predict(X_processed).flatten()
        y_pred = np.expm1(y_pred_log)
        
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"${mae:.2f}")
        col2.metric("MSE", f"{mse:.2f}")
        col3.metric("RMSE", f"${rmse:.2f}")
        col4.metric("R² Score", f"{r2:.4f}")
        
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(y, y_pred, alpha=0.3, s=10)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax.set_xlabel("Actual Price ($)")
        ax.set_ylabel("Predicted Price ($)")
        ax.set_title("Actual vs Predicted Prices")
        st.pyplot(fig)
        
        fig2, ax2 = plt.subplots(figsize=(8,4))
        residuals = y - y_pred
        sns.histplot(residuals, bins=50, kde=True, ax=ax2)
        ax2.axvline(0, color='red', linestyle='--')
        ax2.set_title("Residuals Distribution")
        ax2.set_xlabel("Residual ($)")
        st.pyplot(fig2)
        
    except Exception as e:
        st.error(f"Could not compute performance: {e}")