import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost
from sklearn.preprocessing import MinMaxScaler

# Load the trained XGBoost model (assuming it's saved as 'xgboost_model.json')
model = pickle.load(open('XGBoost_model.pkl', 'rb'))


td = pd.DataFrame({
    'Age': [68, 72, 75],
    'BMI': [28.1, 32.5, 26.8],
    'AlcoholConsumption': [10, 20, 15],
    'PhysicalActivity': [5, 3, 6],
    'DietQuality': [4, 3, 5],
    'SleepQuality': [8, 6, 7],
    'SystolicBP': [130, 140, 125],
    'DiastolicBP': [80, 85, 75],
    'CholesterolTotal': [180, 200, 190],
    'CholesterolLDL': [110, 130, 120],
    'CholesterolHDL': [50, 55, 45],
    'CholesterolTriglycerides': [150, 160, 170],
    'MMSE': [26, 28, 24],
    'FunctionalAssessment': [6, 7, 5],
    'ADL': [8, 9, 7]
})

# Define the columns to be scaled
columns_to_scale = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 
                    'SleepQuality', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 
                    'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 
                    'MMSE', 'FunctionalAssessment', 'ADL']

# Initialize and fit MinMaxScaler
scaler_minmax = MinMaxScaler()
scaler_minmax.fit(td[columns_to_scale])


# Dummy fit for the example (replace with actual saved scaler)
# scaler_minmax.fit(training_data[columns_to_scale])

def preprocess_input(data):
    # Convert categorical inputs to numerical values
    data['Gender'] = 1 if data['Gender'] == 'Male' else 0
    data['Ethnicity'] = 1 if data['Ethnicity'] == 'Ethnic Group 1' else 0
    data['Smoking'] = 1 if data['Smoking'] == 'Yes' else 0
    data['FamilyHistoryAlzheimers'] = 1 if data['FamilyHistoryAlzheimers'] == 'Yes' else 0
    data['CardiovascularDisease'] = 1 if data['CardiovascularDisease'] == 'Yes' else 0
    data['Diabetes'] = 1 if data['Diabetes'] == 'Yes' else 0
    data['Depression'] = 1 if data['Depression'] == 'Yes' else 0
    data['HeadInjury'] = 1 if data['HeadInjury'] == 'Yes' else 0
    data['Hypertension'] = 1 if data['Hypertension'] == 'Yes' else 0
    data['MemoryComplaints'] = 1 if data['MemoryComplaints'] == 'Yes' else 0
    data['BehavioralProblems'] = 1 if data['BehavioralProblems'] == 'Yes' else 0
    data['Confusion'] = 1 if data['Confusion'] == 'Yes' else 0
    data['Disorientation'] = 1 if data['Disorientation'] == 'Yes' else 0
    data['PersonalityChanges'] = 1 if data['PersonalityChanges'] == 'Yes' else 0
    data['DifficultyCompletingTasks'] = 1 if data['DifficultyCompletingTasks'] == 'Yes' else 0
    data['Forgetfulness'] = 1 if data['Forgetfulness'] == 'Yes' else 0

    # Convert the data to a DataFrame
    df = pd.DataFrame([data])
    # Scale the numerical columns
    df[columns_to_scale] = scaler_minmax.transform(df[columns_to_scale])
    return df

def main():
    st.title("Alzheimer's Prediction App")

    # Create form for user input
    with st.form(key='input_form'):
        age = st.number_input("Age", min_value=0, max_value=120, value=77)
        gender = st.selectbox("Gender", ["Male", "Female"])
        ethnicity = st.selectbox("Ethnicity", ["Ethnic Group 1", "Ethnic Group 2"])
        education_level = st.selectbox("Education Level", ["Level 0", "Level 1", "Level 2"])
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=35.55)
        smoking = st.selectbox("Smoking", ["Yes", "No"])
        alcohol_consumption = st.number_input("Alcohol Consumption", min_value=0.0, max_value=100.0, value=15.61)
        physical_activity = st.number_input("Physical Activity", min_value=0.0, max_value=100.0, value=8.53)
        diet_quality = st.number_input("Diet Quality", min_value=0.0, max_value=10.0, value=3.86)
        sleep_quality = st.number_input("Sleep Quality", min_value=0.0, max_value=10.0, value=7.89)
        family_history_alzheimers = st.selectbox("Family History of Alzheimer's", ["Yes", "No"])
        cardiovascular_disease = st.selectbox("Cardiovascular Disease", ["Yes", "No"])
        diabetes = st.selectbox("Diabetes", ["Yes", "No"])
        depression = st.selectbox("Depression", ["Yes", "No"])
        head_injury = st.selectbox("Head Injury", ["Yes", "No"])
        hypertension = st.selectbox("Hypertension", ["Yes", "No"])
        systolic_bp = st.number_input("Systolic BP", min_value=0.0, max_value=300.0, value=114.0)
        diastolic_bp = st.number_input("Diastolic BP", min_value=0.0, max_value=200.0, value=99.72)
        cholesterol_total = st.number_input("Cholesterol Total", min_value=0.0, max_value=500.0, value=187.34)
        cholesterol_ldl = st.number_input("Cholesterol LDL", min_value=0.0, max_value=500.0, value=120.45)
        cholesterol_hdl = st.number_input("Cholesterol HDL", min_value=0.0, max_value=100.0, value=45.63)
        cholesterol_triglycerides = st.number_input("Cholesterol Triglycerides", min_value=0.0, max_value=1000.0, value=278.56)
        mmse = st.number_input("MMSE", min_value=0.0, max_value=30.0, value=9.34)
        functional_assessment = st.number_input("Functional Assessment", min_value=0.0, max_value=10.0, value=5.89)
        memory_complaints = st.selectbox("Memory Complaints", ["Yes", "No"])
        behavioral_problems = st.selectbox("Behavioral Problems", ["Yes", "No"])
        adl = st.number_input("ADL", min_value=0.0, max_value=10.0, value=6.90)
        confusion = st.selectbox("Confusion", ["Yes", "No"])
        disorientation = st.selectbox("Disorientation", ["Yes", "No"])
        personality_changes = st.selectbox("Personality Changes", ["Yes", "No"])
        difficulty_completing_tasks = st.selectbox("Difficulty Completing Tasks", ["Yes", "No"])
        forgetfulness = st.selectbox("Forgetfulness", ["Yes", "No"])

        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        # Gather input data
        input_data = {
            'Age': age,
            'Gender': gender,
            'Ethnicity': ethnicity,
            'EducationLevel': int(education_level.split()[-1]),
            'BMI': bmi,
            'Smoking': smoking,
            'AlcoholConsumption': alcohol_consumption,
            'PhysicalActivity': physical_activity,
            'DietQuality': diet_quality,
            'SleepQuality': sleep_quality,
            'FamilyHistoryAlzheimers': family_history_alzheimers,
            'CardiovascularDisease': cardiovascular_disease,
            'Diabetes': diabetes,
            'Depression': depression,
            'HeadInjury': head_injury,
            'Hypertension': hypertension,
            'SystolicBP': systolic_bp,
            'DiastolicBP': diastolic_bp,
            'CholesterolTotal': cholesterol_total,
            'CholesterolLDL': cholesterol_ldl,
            'CholesterolHDL': cholesterol_hdl,
            'CholesterolTriglycerides': cholesterol_triglycerides,
            'MMSE': mmse,
            'FunctionalAssessment': functional_assessment,
            'MemoryComplaints': memory_complaints,
            'BehavioralProblems': behavioral_problems,
            'ADL': adl,
            'Confusion': confusion,
            'Disorientation': disorientation,
            'PersonalityChanges': personality_changes,
            'DifficultyCompletingTasks': difficulty_completing_tasks,
            'Forgetfulness': forgetfulness
        }

        # Preprocess input data
        input_df = preprocess_input(input_data)

        # Make prediction using the XGBoost model
        prediction = model.predict(input_df)

        # Display the result
        if prediction[0] == 1:
            st.write(f"RESULT: This patient has Alzheimer's")
        else:
            st.write(f"RESULT: This patient does not have Alzheimer's")

if __name__ == '__main__':
    main()
