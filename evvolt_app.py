import streamlit as st
import numpy as np
import joblib

model = joblib.load('EVVolt_model.pkl')  # Load your trained model

st.title('EV Battery Life Prediction')

feature_1 = st.slider('Feature 1', 0.0, 1.0)
feature_2 = st.slider('Feature 2', 0.0, 1.0)
feature_3 = st.slider('Feature 3', 0.0, 1.0)
feature_4 = st.slider('Feature 4', 0.0, 1.0)
feature_5 = st.slider('Feature 5', 0.0, 1.0)

unit = st.selectbox('Select time unit', ('Minutes', 'Hours'))

input_data = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5]])

if st.button('Predict Battery Life'):
    prediction = model.predict(input_data)
    if unit == 'Hours':
        predicted_battery_life = prediction[0] / 60  # Convert minutes to hours
        st.write(f'Predicted Battery Life: {predicted_battery_life:.2f} hours')
    else:
        predicted_battery_life = prediction[0]
        st.write(f'Predicted Battery Life: {predicted_battery_life:.2f} minutes')
