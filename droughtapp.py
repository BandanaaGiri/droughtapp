import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Streamlit page config
st.set_page_config(page_title="Drought Predictor", layout="wide")

# Map numeric labels to descriptive text
LABEL_MAP = {
    0: 'No Drought',
    1: 'Moderate Drought',
    2: 'Severe Drought',
    3: 'Extreme Drought',
    4: 'Exceptional Drought'
}

# Features expected in the input data
FEATURE_COLUMNS = [
    'fips', 'PS', 'T2MDEW', 'T2M_MAX', 'T2M_RANGE', 'TS',
    'WS10M_MIN', 'WS10M_RANGE', 'WS50M_MAX', 'WS50M_RANGE',
    'year', 'lat', 'lon', 'elevation', 'GRS_LAND'
]

# Google Drive file IDs for your model and scaler
SCALER_FILE_ID = "YOUR_SCALER_FILE_ID_HERE"
MODEL_FILE_ID = "YOUR_MODEL_FILE_ID_HERE"

# Paths to save downloaded files locally
SCALER_LOCAL_PATH = "new_scaler.pkl"
MODEL_LOCAL_PATH = "random_forest_model.pkl"

# Download files from Google Drive if not already present
@st.cache_resource(show_spinner="Downloading model and scaler from Google Drive...")
def download_artifacts():
    if not os.path.exists(SCALER_LOCAL_PATH):
        scaler_url = f"https://drive.google.com/uc?id=1ihze7QsGVtzz0tpz1z9Dddi3jczGiPKJusp"
        gdown.download(scaler_url, SCALER_LOCAL_PATH, quiet=False)
    if not os.path.exists(MODEL_LOCAL_PATH):
        model_url = f"https://drive.google.com/uc?id=1WojBceIx8BNrBSC58G6QoJDwnRJ6pNA2"
        gdown.download(model_url, MODEL_LOCAL_PATH, quiet=False)

    scaler = joblib.load(SCALER_LOCAL_PATH)
    model = joblib.load(MODEL_LOCAL_PATH)
    return scaler, model

# Load artifacts
try:
    scaler, model = download_artifacts()
    st.sidebar.success("✅ Model and scaler loaded successfully")
except Exception as e:
    scaler, model = None, None
    st.sidebar.error(f"Failed to load model or scaler: {e}")

# Sidebar navigation menu
page = st.sidebar.selectbox("Navigate", [
    "Upload & Predict",
    "Prediction Results",
    "Confusion Matrix",
    "Feature Importance",
    "Feature Correlation Heatmap"
])

# Helper function to read csv or excel files
def read_file(file):
    try:
        return pd.read_csv(file)
    except Exception:
        return pd.read_excel(file)

# --- Upload & Predict Tab ---
if page == "Upload & Predict":
    st.header("Upload your input data (CSV or Excel)")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file and model and scaler:
        df = read_file(uploaded_file)

        # Check if required columns exist
        missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in uploaded file: {missing_cols}")
            st.stop()
        else:
            st.success("Required columns are present!")

        if st.button("Predict"):
            df_clean = df.dropna(subset=FEATURE_COLUMNS)
            X = df_clean[FEATURE_COLUMNS]
            X_scaled = scaler.transform(X)
            df_clean["prediction"] = model.predict(X_scaled)
            df_clean["drought_label"] = df_clean["prediction"].map(LABEL_MAP)

            st.success("Predictions done!")

            # Store df_clean in session for other tabs
            st.session_state["df"] = df_clean
            st.session_state["predicted"] = True

            # If true labels (column 'score') exist, calculate metrics
            if "score" in df_clean.columns:
                valid = df_clean[df_clean["score"].isin(LABEL_MAP.keys())]
                y_true = valid["score"]
                y_pred = valid["prediction"]

                st.subheader("Evaluation Metrics")
                st.write(f"Accuracy: **{accuracy_score(y_true, y_pred):.4f}**")
                st.write(f"Precision (weighted): **{precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}**")
                st.write(f"Recall (weighted): **{recall_score(y_true, y_pred, average='weighted'):.4f}**")
                st.write(f"F1 Score (weighted): **{f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}**")

                # Save confusion matrix in session
                cm = confusion_matrix(y_true, y_pred, labels=list(LABEL_MAP.keys()))
                st.session_state["confusion_matrix"] = cm
            else:
                st.info("True labels column 'score' not found, skipping evaluation metrics.")

    elif uploaded_file and (model is None or scaler is None):
        st.error("Model and scaler are not loaded properly.")
    else:
        st.info("Upload a data file to start prediction.")

# --- Prediction Results Tab ---
elif page == "Prediction Results":
    st.header("Prediction Results and Filtering")

    if "df" in st.session_state and st.session_state.get("predicted", False):
        df = st.session_state["df"]

        selected_label = st.selectbox("Filter by Drought Severity", ["All"] + list(LABEL_MAP.values()))

        if selected_label != "All":
            filtered = df[df["drought_label"] == selected_label]
        else:
            filtered = df

        st.dataframe(filtered[[*FEATURE_COLUMNS, "prediction", "drought_label"]].head(50))

        st.download_button(
            label="Download Filtered Results as CSV",
            data=filtered.to_csv(index=False).encode(),
            file_name="drought_predictions.csv",
            mime="text/csv"
        )
    else:
        st.info("Run predictions first in 'Upload & Predict' tab.")

# --- Confusion Matrix Tab ---
elif page == "Confusion Matrix":
    st.header("Confusion Matrix")

    if "confusion_matrix" in st.session_state:
        cm = st.session_state["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=[LABEL_MAP[i] for i in LABEL_MAP], columns=[LABEL_MAP[i] for i in LABEL_MAP])

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)
    else:
        st.warning("No confusion matrix available. Run predictions with true labels first.")

# --- Feature Importance Tab ---
elif page == "Feature Importance":
    st.header("Feature Importance")

    if model and hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
        fig, ax = plt.subplots(figsize=(10, 6))
        importances.plot(kind='barh', ax=ax, color='teal')
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    else:
        st.warning("Model does not provide feature importance.")

# --- Feature Correlation Heatmap Tab ---
elif page == "Feature Correlation Heatmap":
    st.header("Feature Correlation Heatmap")

    if "df" in st.session_state:
        df = st.session_state["df"]
        corr = df[FEATURE_COLUMNS].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap of Features")
        st.pyplot(fig)
    else:
        st.info("Run prediction first to see correlation heatmap.")

# --- Feedback and footer ---
st.sidebar.markdown("---")
st.sidebar.subheader("Feedback")
feedback = st.sidebar.text_area("Your feedback")
if st.sidebar.button("Submit"):
    st.sidebar.success("Thanks for your feedback!")

st.markdown("---")
st.markdown("© 2025 | Developed by Bandana Giri | Version 1.0")
