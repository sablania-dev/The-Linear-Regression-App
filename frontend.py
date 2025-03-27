import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt  # Ensure this import exists
from regression_backend import RegressionAnalysis
from ydata_profiling import ProfileReport  # Import pandas profiling
from streamlit.components.v1 import html  # For embedding HTML

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
        
        # Ensure the dataset has at least two columns (features and target)
        if len(df.columns) < 2:
            st.error("The dataset must have at least one feature column and one target column.")
            return
        
        # Add a button to generate the pandas profiling report
        if st.button("Generate Pandas Profiling Report"):
            # Generate pandas profiling report
            profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
            report_path = os.path.join("temp", "pandas_profiling_report.html")
            profile.to_file(report_path)
            
            # Provide a download button for the report
            with open(report_path, "rb") as report_file:
                report_bytes = report_file.read()
                st.download_button(
                    label="Download Pandas Profiling Report",
                    data=report_bytes,
                    file_name="pandas_profiling_report.html",
                    mime="text/html"
                )
        
        missing_value_option = st.selectbox("Handle Missing Values", ["None", "Drop Rows", "Fill with Mean", "Fill with Median"])
        scale_data = st.checkbox("Apply Feature Scaling")
        polynomial_degree = st.slider("Polynomial Degree", min_value=1, max_value=5, value=1)
        model_type = st.selectbox("Select Regression Model", ["linear", "ridge", "lasso"])
        
        outlier_removal = st.checkbox("Remove Outliers (Z-score Method)")
        
        if st.button("Run Regression Analysis"):
            reg = RegressionAnalysis(file_path, scale_data, polynomial_degree, model_type, missing_value_option, outlier_removal)
            results = reg.run_analysis()  # Capture the returned results
            
            st.write("### Regression Results")
            # Display key metrics as a table with separate columns for training and test sets
            metrics = {
                "Metric": ["R2", "Adjusted R2", "MSE", "MAE", "RMSE"],
                "Training Set": [
                    results["R2_train"], results["Adjusted_R2_train"],
                    results["MSE_train"], results["MAE_train"], results["RMSE_train"]
                ],
                "Test Set": [
                    results["R2_test"], results["Adjusted_R2_test"],
                    results["MSE_test"], results["MAE_test"], results["RMSE_test"]
                ]
            }
            st.table(pd.DataFrame(metrics))  # Display metrics as a table
            
            # Show best parameters if the model is ridge or lasso
            if model_type in ["ridge", "lasso"] and "best_params" in results:
                st.write("### Best Parameters")
                best_params_df = pd.DataFrame({
                    "Parameter": list(results["best_params"].keys()),
                    "Value": list(results["best_params"].values())
                })
                st.table(best_params_df)  # Display the best parameters as a table
            
            if "coefficients" in results:
                st.write("### Model Coefficients")
                coefficients = pd.DataFrame({
                    "Coefficient": results["coefficients"].index,
                    "Value": results["coefficients"].values,
                    "Std Error": results["std_errors"].values,
                    "P-Value": results["p_values"].values
                })
                st.dataframe(coefficients)  # Display coefficients as a table
            
            if model_type in ["ridge", "lasso"]:
                st.write("### Training vs Validation Loss Plot")
                st.image(results['tuning_plot_path'])  # Display the saved tuning plot
            
            st.write("### Residual Plot")
            st.image(os.path.join("images", "residuals_plot.png"))  # Load the saved residuals plot

if __name__ == "__main__":
    main()
