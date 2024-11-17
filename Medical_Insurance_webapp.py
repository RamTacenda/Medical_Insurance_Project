# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:29:17 2024

@author: Dell
"""

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import streamlit as st


# Loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
normalizer = pickle.load(open('scaler.pkl', 'rb'))

# Creating the function
def prediction(input_data):
    if(input_data[0] == 0): return (f"Predicted Charges: RS: 0.00")
    mapp = {"Northeast": [1.0, 0.0, 0.0, 0.0],
            "Northwest": [0.0, 1.0, 0.0, 0.0],
            "Southeast": [0.0, 0.0, 1.0, 0.0],
            "Southwest": [0.0, 0.0, 0.0, 1.0]}
    final = input_data[:3]
    if(input_data[3] == "Female"):
        final.extend([0.0, 1.0])
    else:
        final.extend([1.0, 0.0])
    
    if(input_data[4] == "Yes"):
        final.extend([0.0, 1.0])
    else:
        final.extend([1.0, 0.0])
    
    final.extend(mapp[input_data[-1]])
        
    input_data_numpy_arr = np.asarray(final)
    input_data_reshaped = input_data_numpy_arr.reshape(1,-1)
    input_data_norm = normalizer.transform(input_data_reshaped)
    return (f"Predicted Charges: RS: {loaded_model.predict(input_data_norm)[0]: .2f}")
    


# Main function
def main():
    # App title with background color
    
    st.markdown(
        """
        <div style="background-color:#f44336;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">Medical Insurance Cost Prediction</h1>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("### Please provide the following details:")

    # Divide input into two columns for better alignment
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, step=1)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=18.0, step=0.1, format="%.2f")
        children = st.number_input("Number of Children", min_value=0, step=1)
    
    with col2:
        sex = st.selectbox("Sex", options=["Male", "Female"])
        smoker = st.selectbox("Smoker", options=["Yes", "No"])
        region = st.selectbox("Region", options=["Northeast", "Northwest", "Southeast", "Southwest"])
    
    # Add a prediction button with visual feedback
    if st.button("Predict", help="Click to calculate the insurance cost"):
        with st.spinner("Calculating..."):
            calculated = prediction([age, bmi, children, sex, smoker, region])
        st.success("Prediction Completed!")
        st.info(calculated, icon="💰")
    else:
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing for alignment

    # Footer section with contact/info
    st.markdown(
        """
        <hr>
        <div style="text-align:center;">
        <p style="color:gray;">Model Created and deployed by RamTacenda 🤍</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
if __name__ == '__main__':
    main()