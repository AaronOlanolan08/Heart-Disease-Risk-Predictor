import streamlit as st
import joblib
import pandas as pd

# Load model and pipeline
model = joblib.load('models/model.joblib')
pipeline = joblib.load('models/preprocessing.joblib')

st.title("Heart Disease Risk Prediction")
st.write("Enter the patient's information below:")

# User input in the Streamlit components
def user_input():
    Sex = st.radio('Sex', ['Male', 'Female'])
    AgeCategory = st.selectbox('Age Category', [
        '18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
        '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'
    ])
    Race = st.selectbox('Race', [
        'White', 'Black', 'Asian', 'American Indian/Alaskan Native',
        'Hispanic', 'Other'
    ])
    Diabetic = st.selectbox(
        'Diabetic', ['Yes', 'No', 'No, borderline diabetes', 'Yes (during pregnancy)'])
    GenHealth = st.selectbox(
        'General Health', ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'])
    BMI = st.number_input('BMI', 10.0, 50.0, step=0.1)
    PhysicalHealth = st.slider(
        'Number of days in a month an individual experienced poor physical health', 0, 30, 0)
    MentalHealth = st.slider(
        'Number of days in a month an individual experienced poor mental health', 0, 30, 0)
    SleepTime = st.slider('Sleep Time (hours)', 1, 24, 7)

    # Binary features
    Smoking = st.checkbox('Smokes?')
    AlcoholDrinking = st.checkbox('Drinks Alcohol?')
    Stroke = st.checkbox('History of Stroke?')
    DiffWalking = st.checkbox('Difficulty Walking?')
    PhysicalActivity = st.checkbox('Physically Active?')
    Asthma = st.checkbox('Has Asthma?')
    KidneyDisease = st.checkbox('Kidney Disease?')
    SkinCancer = st.checkbox('Skin Cancer?')

    # Pass the data
    data = {
        'BMI': BMI,
        'PhysicalHealth': PhysicalHealth,
        'MentalHealth': MentalHealth,
        'SleepTime': SleepTime,
        'AgeCategory': AgeCategory,
        'Race': Race,
        'Diabetic': Diabetic,
        'GenHealth': GenHealth,
        'Smoking': int(Smoking),
        'AlcoholDrinking': int(AlcoholDrinking),
        'Stroke': int(Stroke),
        'DiffWalking': int(DiffWalking),
        'Sex': 1 if Sex.lower() == 'male' else 0,
        'PhysicalActivity': int(PhysicalActivity),
        'Asthma': int(Asthma),
        'KidneyDisease': int(KidneyDisease),
        'SkinCancer': int(SkinCancer)
    }

    return pd.DataFrame([data])


input_df = user_input()

# Process the data in the pipeline and model 
if st.button('Predict'):
    processed = pipeline.transform(input_df)
    prediction = model.predict(processed)[0]
    proba = model.predict_proba(processed)[0][1]

    # Print results
    result = "AT RISK of Heart Disease" if prediction == 1 else "NOT AT RISK"
    st.subheader("Prediction: ")

    if proba >= 0.80:
        st.error(f"Confidence: {proba:.2%} - {result} (High risk)")
    elif proba >= 0.60:
        st.error(f"Confidence: {proba:.2%} - {result} (Moderate risk)")
    elif proba >=0.50:
        st.error(f"Confidence: {proba:.2%} - {result} (Low risk)")
    else:
        st.success(f"Confidence: {proba:.2%} - {result}")
