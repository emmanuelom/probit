
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import matplotlib.pyplot as plt

# Streamlit title
st.title("Probit Analysis for Dose-Response Data")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a pandas dataframe
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

        # Log-transform the dose, handling zeros carefully
        data['log_dose'] = np.log10(data['Dose'].replace(0, np.nan))
        data = data.dropna()  # Drop rows where log of dose is undefined (i.e., dose = 0)

        # Display processed data
        st.subheader("Processed Data (with log-transformed Dose)")
        st.write(data)

        # Define the independent (X) and dependent (y) variables
        X = data['log_dose']
        y = data['p']

        # Add constant to X for intercept
        X_with_constant = sm.add_constant(X)

        # Fit the Probit model
        model = Probit(y, X_with_constant)
        result = model.fit()

        # Display model summary
        st.subheader("Probit Model Summary")
        st.text(result.summary())

        # Predict response probabilities
        data['predicted_p'] = result.predict(X_with_constant)

        # Display the data with predicted probabilities
        st.subheader("Data with Predicted Probabilities")
        st.write(data)

        # Calculate DE50 (dose at which 50% of the population responds)
        DE50_log_dose = -result.params[0] / result.params[1]  # Solving for log dose where predicted_p = 0.5
        DE50_dose = 10 ** DE50_log_dose  # Convert back to dose from log10

        # Display DE50 result
        st.subheader(f"Estimated DE50 (dose at 50% response): {DE50_dose:.2f} mg/mL")

        # Plot the observed proportions vs. log dose and the fitted Probit curve
        st.subheader("Probit Regression Plot")

        # Set up the figure
        fig, ax = plt.subplots()

        # Scatter plot for observed data (actual proportions)
        ax.scatter(data['log_dose'], data['p'], color='blue', label='Observed Proportions', zorder=3)

        # Generate fitted values for a range of doses
        log_dose_range = np.linspace(data['log_dose'].min(), data['log_dose'].max(), 100)
        X_range_with_constant = sm.add_constant(log_dose_range)
        fitted_p = result.predict(X_range_with_constant)

        # Plot the fitted Probit curve
        ax.plot(log_dose_range, fitted_p, color='red', label='Fitted Probit Curve', zorder=2)

        # Add a vertical line for DE50
        ax.axvline(x=DE50_log_dose, color='green', linestyle='--', label=f'DE50 = {DE50_dose:.2f} mg/mL', zorder=1)

        # Labeling the plot
        ax.set_xlabel("Log Dose (log10 mg/mL)")
        ax.set_ylabel("Proportion Responding")
        ax.set_title("Probit Regression and Estimated DE50")
        ax.legend()

        # Display the plot
        st.pyplot(fig)
