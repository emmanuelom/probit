import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm

# Streamlit title
st.title("Probit Score Calculation with Bias Term")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(uploaded_file)

    # Check if the required columns are present
    required_columns = ['Dose', 'n', 'r']
    if not all(col in data.columns for col in required_columns):
        st.error(f"CSV file must contain the following columns: {', '.join(required_columns)}")
    else:
        # Show the uploaded data
        st.subheader("Uploaded Data")
        st.write(data)

        # Calculate the proportion of responders (r/n)
        data['p'] = data['r'] / data['n']
        
        # Create a slider for the bias term
        bias = st.slider("Select Bias Term", min_value=0.0, max_value=10.0, value=5.0)

        # Calculate Probit scores using the cumulative distribution function and add bias
        data['Probit Score'] = norm.ppf(data['p']) + bias

        # Log-transform the dose for regression
        data['log_dose'] = np.log10(data['Dose'].replace(0, np.nan))
        data = data.dropna(subset=['log_dose', 'Probit Score'])  # Drop rows with NaN values

        # Display the data with Probit scores
        st.subheader("Data with Probit Scores")
        st.write(data[['Dose', 'log_dose','n', 'r', 'p', 'Probit Score']])

        # Show a message if any probabilities are out of range
        if (data['p'] <= 0).any() or (data['p'] >= 1).any():
            st.warning("Proportions must be between 0 and 1. Probit scores for these values may be NaN.")
