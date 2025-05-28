
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF

model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

feature_columns = [
    'Age', 'Gender', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
    'nausea', 'joint_pain', 'abdominal_pain', 'high_fever', 'chills',
    'fatigue', 'runny_nose', 'pain_behind_the_eyes', 'dizziness', 'headache',
    'chest_pain', 'vomiting', 'cough', 'shivering', 'asthma_history',
    'high_cholesterol', 'diabetes', 'obesity', 'hiv_aids', 'nasal_polyps',
    'asthma', 'high_blood_pressure', 'severe_headache', 'weakness',
    'trouble_seeing', 'fever', 'body_aches', 'sore_throat', 'sneezing',
    'diarrhea', 'rapid_breathing', 'rapid_heart_rate', 'pain_behind_eyes',
    'swollen_glands', 'rashes', 'sinus_headache', 'facial_pain',
    'shortness_of_breath', 'reduced_smell_and_taste', 'skin_irritation',
    'itchiness', 'throbbing_headache', 'confusion', 'back_pain', 'knee_ache'
]

numeric_cols = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)']
symptom_cols = feature_columns[5:]

st.set_page_config(page_title="Weather-Driven Disease Prediction", layout="wide")
st.title("Weather-Driven Disease Prediction Using Machine Learning Models")

if "auto_fill" not in st.session_state:
    st.session_state.auto_fill = False
if "filled_symptoms" not in st.session_state:
    st.session_state.filled_symptoms = {}
if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("### Patient Info")
    if st.session_state.auto_fill:
        age = st.number_input("Age", 0, 120, value=45)
        gender = st.radio("Gender", ["Male", "Female"], index=1)
        temp = st.slider("Temperature (°C)", 30.0, 45.0, value=38.5)
        humidity = st.slider("Humidity (%)", 10.0, 100.0, value=85.0)
        wind_speed = st.slider("Wind Speed (km/h)", 0.0, 100.0, value=12.0)
    else:
        age = st.number_input("Age", 0, 120, 30)
        gender = st.radio("Gender", ["Male", "Female"])
        temp = st.slider("Temperature (°C)", 30.0, 45.0, 36.5)
        humidity = st.slider("Humidity (%)", 10.0, 100.0, 60.0)
        wind_speed = st.slider("Wind Speed (km/h)", 0.0, 100.0, 5.0)

    st.markdown("### Symptoms")
    symptom_state = {}
    grid_cols = st.columns(3)
    for i, symptom in enumerate(symptom_cols):
        col = grid_cols[i % 3]
        default_val = st.session_state.filled_symptoms.get(symptom, False) if st.session_state.auto_fill else False
        symptom_state[symptom] = col.checkbox(symptom, value=default_val)

    if st.button("Auto-fill Example"):
        st.session_state.auto_fill = True
        st.session_state.filled_symptoms = {s: np.random.rand() > 0.85 for s in symptom_cols}
        st.rerun()

with col2:
    st.markdown("### Prediction & Insight")

    if st.button("Run Prediction"):
        gender_val = 0 if gender == "Male" else 1
        input_data = [age, gender_val, temp, humidity, wind_speed]
        for symptom in symptom_cols:
            input_data.append(1 if symptom_state[symptom] else 0)

        df_input = pd.DataFrame([input_data], columns=feature_columns)
        df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])
        probs = model.predict_proba(df_input)[0]
        top_indices = np.argsort(probs)[::-1][:3]
        top_labels = label_encoder.inverse_transform(top_indices)
        top_probs = probs[top_indices]

        st.success(f"Top Prediction: {top_labels[0]} ({top_probs[0]*100:.1f}%)")

        fig, ax = plt.subplots()
        ax.barh(top_labels[::-1], top_probs[::-1]*100, color='#5DADE2')
        ax.set_xlabel("Probability (%)")
        ax.set_title("Top 3 Predicted Diseases")
        st.pyplot(fig)

        selected_symptoms = [s for s, selected in symptom_state.items() if selected]
        insight = f"Based on symptoms like {', '.join(selected_symptoms[:3])}, the model predicts {top_labels[0]} with highest confidence."
        st.markdown(f"Insight: {insight}")

        record = {
            "Age": age, "Gender": gender, "Temp": temp, "Humidity": humidity,
            "Wind": wind_speed, "Top Prediction": top_labels[0],
            "Top 3": ", ".join(top_labels), "Top Prob": f"{top_probs[0]*100:.1f}%",
            "Insight": insight
        }
        st.session_state.history.append(record)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Health Prediction Report", ln=True, align='C')
        pdf.ln(10)
        for k, v in record.items():
            pdf.multi_cell(0, 10, f"{k}: {v}")
        pdf_output = pdf.output(dest='S').encode('latin1')

        st.download_button("Download PDF", data=pdf_output,
                           file_name="prediction_report.pdf", mime="application/pdf")

    if st.session_state.history:
        st.markdown("### Prediction History")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df)

        csv = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "prediction_history.csv", "text/csv")

st.markdown("---")
st.header("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])
if uploaded_file:
    try:
        batch_df = pd.read_csv(uploaded_file)
        batch_df[numeric_cols] = scaler.transform(batch_df[numeric_cols])
        preds = model.predict(batch_df)
        batch_df["Prediction"] = label_encoder.inverse_transform(preds)
        st.dataframe(batch_df)

        result_csv = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Batch Results", result_csv, "batch_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error processing file: {e}")
