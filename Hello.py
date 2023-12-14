# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import RobustScaler
import pickle
import tensorflow as tf
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

def run():
    st.title("Stroke Prediction Project")
    st.subheader("Add Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=0.0, format="%.2f")
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])

    with col2:
        avg_glucose_level = st.number_input(
            "Average Glucose Level", min_value=0.0, format="%.2f")
        bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
        work_type = st.selectbox(
            "Work Type", ["Govt Job", "Never Worked", "Private", "Self-employed", "Children"])
        smoking_status = st.selectbox(
            "Smoking Status", ["Unknown", "Formerly Smoked", "Never Smoked", "Smokes"])
        residence_type = st.selectbox("Residence Type", ["Rural", "Urban"])

    gender = 1 if gender == "Male" else 0
    hypertension = 1 if hypertension == "Yes" else 0
    heart_disease = 1 if heart_disease == "Yes" else 0
    ever_married = 1 if ever_married == "Yes" else 0
    residence_type = 1 if residence_type == "Urban" else 0

    work_type_mapping = {
        "Govt Job": 0,
        "Never Worked": 1,
        "Private": 2,
        "Self-employed": 3,
        "Children": 4
    }

    s_s_mapping = {
        "Unknown": 0,
        "Formerly Smoked": 1,
        "Never Smoked": 2,
        "Smokes": 3
    }

    work_type_encoded = np.zeros(5)
    work_type_encoded[work_type_mapping[work_type]] = 1

    s_s_encoded = np.zeros(4)
    s_s_encoded[s_s_mapping[smoking_status]] = 1

    patient_details = np.array([gender, age, hypertension, heart_disease,
                            ever_married, residence_type, avg_glucose_level, bmi])
    patient_details = np.concatenate(
        (patient_details, work_type_encoded, s_s_encoded))

    if st.button("Predict Stroke"):
        input_data = {
            "gender": [gender],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "ever_married": [ever_married],
            "Residence_type": [residence_type],
            "avg_glucose_level": [avg_glucose_level],
            "bmi": [bmi],
            "work_type_Govt_job": [work_type_encoded[0]],
            "work_type_Never_worked": [work_type_encoded[1]],
            "work_type_Private": [work_type_encoded[2]],
            "work_type_Self-employed": [work_type_encoded[3]],
            "work_type_children": [work_type_encoded[4]],
            "s_s_Unknown": [s_s_encoded[0]],
            "s_s_formerly smoked": [s_s_encoded[1]],
            "s_s_never smoked": [s_s_encoded[2]],
            "s_s_smokes": [s_s_encoded[3]]
        }
        input_df = pd.DataFrame(input_data)
        
        scaler_path = "scalerfile.pickle"
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        input_scaled = scaler.transform(input_df)

        model_path = "GRU_Robust_SMOTE_modelfile.h5"
        model = tf.keras.models.load_model(model_path)
        input_scaled = input_scaled.reshape(
            input_scaled.shape[0], input_scaled.shape[1], 1)

        predictions = model.predict(input_scaled)
        
        print('Prediction: ', predictions)
        prediction = 1 if predictions[0][0] > 0.1 else 0
        
        # st.write("Predicted Stroke Category:", prediction)
        st.subheader("Possibility of Stroke")
        # st.progress(predictions[0][0]/2*10)
        if prediction == 1:
            st.error("POSSIBILITY OF STROKE ☹️")
        else:
            st.success("OUT OF DANGER 😀")


if __name__ == "__main__":
    run()
