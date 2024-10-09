import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
#import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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

        # Filter out rows where p is 0 or 1, to avoid Probit score being infinity or NaN
        data = data[(data['p'] > 0) & (data['p'] < 1)]
        
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

        # Fit the linear regression model using sklearn
        X = data['log_dose'].values.reshape(-1, 1)  # Reshape X to a 2D array
        y = data['Probit Score'].values
        model = LinearRegression()
        model.fit(X, y)

        # Get the regression coefficients and intercept
        intercept = model.intercept_ #alpha
        slope = model.coef_[0] # beta

        # Display the linear regression equation
        st.subheader("Linear Regression Equation (y_i = alpha + beta * x_i)")
        st.write(f"Probit Score = {intercept:.3f} + {slope:.3f} * Log Dose")
        

        # Get mu & sigma from LR
        mu = (bias - intercept)/slope
        sigma = 1/slope
        cl_50 = 10**mu
        st.write(f"Probit mu = {mu:.3f}, sigma = {sigma:.3f}, CL_50 = {cl_50:.3f} mg/mL")

        # User input for Dose
        dose_input = st.text_input("Enter Dose to compute Probit Score:", value="1")

        if dose_input:
            try:
                dose_value = float(dose_input)

                if dose_value <= 0:
                    st.error("Dose value must be greater than 0.")
                else:
                    # Log-transform the entered dose
                    log_dose_input = np.log10(dose_value)

                    # Compute the corresponding Probit score using the linear regression model
                    predicted_probit_score = model.predict([[log_dose_input]])[0]

                    # Display the result
                    st.write(f"Predicted Probit Score for Dose {dose_value} mg/mL: {predicted_probit_score:.3f}")
            except ValueError:
                st.error("Invalid input. Please enter a numeric value for the dose.")


        # Plotting the regression line
        st.subheader("Linear Regression Plot")
        fig, ax = plt.subplots()

        # Scatter plot for observed data
        ax.scatter(data['log_dose'], data['Probit Score'], color='blue', label='Observed Probit Scores')

        # Plot the regression line
        ax.plot(data['log_dose'], model.predict(X), color='red', label='Regression Line')

        # Add labels and title
        ax.set_xlabel("Log Dose (log10 mg/mL)")
        ax.set_ylabel("Probit Score")
        ax.set_title("Linear Regression of Probit Scores vs. Log Dose")
        ax.legend()

        # Display the plot
        st.pyplot(fig)

        