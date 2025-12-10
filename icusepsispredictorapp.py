import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# --- Configuration ---
DATA_FILE_PATH = 'sepsis_input_data.csv'
MODEL_FEATURES = ['Age', 'BP_Systolic', 'Pulse_bpm', 'TLC', 'Serum_Creatinine', 'Total_Bilirubin']

# --- Page Setup ---
st.set_page_config(
    page_title="ICU Sepsis Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title and Description ---
st.title("🏥 ICU Sepsis Risk Predictor")
st.markdown("""
This application uses a Logistic Regression model trained on clinical data to estimate a patient's risk of Sepsis based on their vital signs and lab values.
""")

# --- Load and Preprocess Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        # Drop rows with any missing data in the model features (for simplicity)
        df.dropna(subset=MODEL_FEATURES + ['Sepsis_Label'], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: The data file '{DATA_FILE_PATH}' was not found. Ensure it is in the root folder.")
        return pd.DataFrame()

df = load_data()

# --- Model Training (Train/Test Split and Scaling) ---
if not df.empty:
    X = df[MODEL_FEATURES]
    y = df['Sepsis_Label']

    # Splitting data for internal validation (optional, but good practice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Note: X_test is not used for this simple app, but the train/test split is kept for best practice.

    # Training the model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    st.sidebar.header("Model Performance")
    # Provide a dummy performance metric since the data is small
    y_pred_test = model.predict(scaler.transform(X_test))
    report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
    st.sidebar.metric(label="Model Accuracy (Test Data)", value=f"{report['accuracy']:.2f}")


    # --- User Input Sidebar ---
    st.sidebar.header("Patient Input")

    # The input fields use the High/Low risk values as a guide for the user
    age = st.sidebar.slider('Age (Years)', min_value=18, max_value=90, value=65)
    
    bp_systolic = st.sidebar.slider('Systolic BP (mmHg)', min_value=70, max_value=150, value=110)
    st.sidebar.info("Low BP (≤100) is a high-risk qSOFA indicator.")

    pulse_bpm = st.sidebar.slider('Pulse Rate (bpm)', min_value=40, max_value=140, value=95)
    st.sidebar.info("High Pulse (>90) is a high-risk SIRS indicator.")

    tlc = st.sidebar.number_input('Total Leukocyte Count (TLC) (x10^3/μL)', min_value=1.0, max_value=40.0, value=10.1, step=0.1)
    st.sidebar.info("TLC <4 or >12 are high-risk SIRS indicators.")

    creatinine = st.sidebar.number_input('Serum Creatinine (mg/dL)', min_value=0.1, max_value=5.0, value=0.9, step=0.1)
    st.sidebar.info("Creatinine ≥2.0 is a high-risk SOFA indicator (Kidney).")

    bilirubin = st.sidebar.number_input('Total Bilirubin (mg/dL)', min_value=0.1, max_value=10.0, value=0.6, step=0.1)
    st.sidebar.info("Bilirubin ≥2.0 is a high-risk SOFA indicator (Liver).")

    # Create the input dataframe
    input_data = pd.DataFrame([[age, bp_systolic, pulse_bpm, tlc, creatinine, bilirubin]], 
                              columns=MODEL_FEATURES)

    # --- Prediction Button and Logic ---
    if st.sidebar.button("Predict Sepsis Risk"):
        # Scale the user input
        input_scaled = scaler.transform(input_data)
        
        # Predict probability
        prediction_proba = model.predict_proba(input_scaled)[0][1] * 100
        
        st.subheader("Prediction Result")
        
        if prediction_proba >= 50:
            st.error(f"**HIGH RISK** of Sepsis or Poor Outcome: {prediction_proba:.2f}%")
            st.markdown("🚨 **Recommendation:** Immediately calculate full SOFA score, perform blood cultures, and begin Sepsis-management protocols (e.g., the 'Sepsis Bundle').")
        else:
            st.success(f"**LOW RISK** of Sepsis or Poor Outcome: {prediction_proba:.2f}%")
            st.markdown("✅ **Recommendation:** Continue close monitoring, especially if clinical suspicion remains high. Re-evaluate vital signs frequently.")
            
        st.write("---")
        st.subheader("Patient Input Vitals")
        st.dataframe(input_data)

else:
    st.error("The app could not load the data. Please check 'sepsis_input_data.csv'.")
