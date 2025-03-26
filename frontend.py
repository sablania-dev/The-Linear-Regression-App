import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt  # Ensure this import exists
from regression_backend import RegressionAnalysis

def main():
    st.title("Regression Analysis Tool")
    
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xls", "xlsx"])
    
    if uploaded_file is not None:
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
        st.write("### Data Preview")
        st.write(df.head())
        
        scale_data = st.checkbox("Apply Feature Scaling")
        polynomial_degree = st.slider("Polynomial Degree", min_value=1, max_value=5, value=1)
        model_type = st.selectbox("Select Regression Model", ["linear", "ridge", "lasso"])
        
        missing_value_option = st.selectbox("Handle Missing Values", ["None", "Drop Rows", "Fill with Mean", "Fill with Median"])
        outlier_removal = st.checkbox("Remove Outliers (Z-score Method)")
        
        if st.button("Run Regression Analysis"):
            reg = RegressionAnalysis(file_path, scale_data, polynomial_degree, model_type, missing_value_option, outlier_removal)
            results = reg.run_analysis()  # Capture the returned results
            
            st.write("### Regression Results")
            st.json(results)  # Display the results
            
            st.write("### Distribution Plots")
            st.image(os.path.join("images", "distribution_plots.png"))  # Load the saved distribution plots
            
            st.write("### Residual Plot")
            st.image(os.path.join("images", "residuals_plot.png"))  # Load the saved residuals plot

if __name__ == "__main__":
    main()
