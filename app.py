import streamlit as st
import joblib
import pandas as pd


model = joblib.load('XGBoost_heart.pkl')
scaler = joblib.load('scaler.pkl')
expeted_columns = joblib.load('columns.pkl')


st.title('Heart Disease Prediction by Ankit')
st.markdown('This is a simple web application that predicts the likelihood of heart disease based on user input. Please fill in the required information below and click the "Predict" button to see your results.')

age = st.number_input('Age', min_value=18, max_value=100, value=40)
sex = st.selectbox('Sex', options=['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', options=['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120)    
chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=400, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options   =['Yes', 'No'])     
restecg = st.selectbox('Resting Electrocardiographic Results', options=['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina', options=['Yes', 'No'])
oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=['Upsloping', 'Flat', 'Downsloping'])


if st.button('Predict'):
    raw_input = {
    'age': age,
    'sex': 1 if sex == 'Male' else 0,
    'cp': ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'].index(cp),
    'trestbps': trestbps,                               
    'chol': chol,
    'fbs': 1 if fbs == 'Yes' else 0,
    'restecg': ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'].index(restecg),
    'thalach': thalach,
    'exang': 1 if exang == 'Yes' else 0,
    'oldpeak': oldpeak,
    'slope': ['Upsloping', 'Flat', 'Downsloping'].index(slope)
    }

    input_df = pd.DataFrame([raw_input])
    for col in expeted_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[expeted_columns]
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error('⚠️ Based on the provided information, there is a high likelihood of heart disease. Please consult a healthcare professional for further evaluation and advice.') 
    else:
         st.success('✅ Based on the provided information, there is a low likelihood of heart disease. However, it is always recommended to maintain a healthy lifestyle and consult a healthcare professional for regular check-ups.')