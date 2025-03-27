import streamlit as st
import pandas as pd
import os
from regression_backend import RegressionAnalysis
from ydata_profiling import ProfileReport  # Import pandas profiling

def generate_profiling_report(df):
    """Generate and provide a download button for the pandas profiling report."""
    profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    report_path = os.path.join("temp", "pandas_profiling_report.html")
    profile.to_file(report_path)
    with open(report_path, "rb") as report_file:
        report_bytes = report_file.read()
        st.download_button(
            label="Download Pandas Profiling Report",
            data=report_bytes,
            file_name="pandas_profiling_report.html",
            mime="text/html"
        )

def display_regression_results(results, model_type):
    """Display regression results, including metrics, best parameters, and coefficients."""
    st.write("### Regression Results")
    metrics = {
        "Metric": ["R2", "Adjusted R2", "MSE", "MAE", "RMSE"],
        "Training Set": [
            results["R2_train"], results["Adjusted_R2_train"],
            results["MSE_train"], results["MAE_train"], results["RMSE_train"]
        ],
        "Validation Set": [
            results["R2_val"], results["Adjusted_R2_val"],
            results["MSE_val"], results["MAE_val"], results["RMSE_val"]
        ],
        "Test Set": [
            results["R2_test"], results["Adjusted_R2_test"],
            results["MSE_test"], results["MAE_test"], results["RMSE_test"]
        ]
    }
    st.table(pd.DataFrame(metrics))  # Display metrics as a table

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

def handle_file_upload():
    """Handle file upload and return the uploaded DataFrame and file path."""
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
        return df, file_path
    return None, None

def select_features_and_target(df):
    """Allow users to select the target variable and input features."""
    target_column = st.selectbox("Select Target Variable", options=list(df.columns), index=len(df.columns) - 1)
    available_features = [col for col in df.columns if col != target_column]
    feature_columns = st.multiselect(
        "Select Input Features",
        options=available_features,
        default=available_features
    )
    if not feature_columns:
        st.error("Please select at least one input feature.")
        return None, None
    return feature_columns, target_column

def filter_dataset(df, feature_columns, target_column, file_path):
    """Filter the dataset to include only selected features and the target variable."""
    filtered_df = df[feature_columns + [target_column]]
    filtered_file_path = os.path.join("temp", "filtered_" + os.path.basename(file_path))
    filtered_df.to_excel(filtered_file_path, index=False, engine='openpyxl')
    return filtered_df, filtered_file_path

def display_column_statistics(preprocessed_data):
    """Display statistics (min, avg, max, etc.) of the preprocessed and transformed columns."""
    st.write("### Column Statistics (After Transformation)")
    stats = pd.DataFrame(preprocessed_data).describe().T  # Transpose for better readability
    stats["median"] = pd.DataFrame(preprocessed_data).median()  # Add median to the statistics
    st.dataframe(stats)  # Display the statistics as a table

def main():
    st.title("Regression Analysis Tool")
    
    df, file_path = handle_file_upload()
    if df is not None:
        st.write("### Data Preview")
        st.write(df.head())
        
        if len(df.columns) < 2:
            st.error("The dataset must have at least one feature column and one target column.")
            return
        
        feature_columns, target_column = select_features_and_target(df)
        if feature_columns is None or target_column is None:
            return
        
        df, file_path = filter_dataset(df, feature_columns, target_column, file_path)
        
        if st.button("Generate Pandas Profiling Report"):
            generate_profiling_report(df)
        
        missing_value_option = st.selectbox(
            "Handle Missing Values",
            ["None", "Drop Rows", "Fill with Mean", "Fill with Median", "MICE Imputer", "KNN Imputer"]
        )
        scaling_option = st.selectbox("Feature Scaling", ["None", "Min-Max Scaling", "Standard Scaling"])
        polynomial_degree = st.slider("Polynomial Degree", min_value=1, max_value=5, value=1)
        model_type = st.selectbox("Select Regression Model", ["linear", "ridge", "lasso"])
        outlier_removal = st.checkbox("Remove Outliers (Z-score Method)")
        
        # Preprocess the data to apply transformations
        scale_data = scaling_option != "None"
        reg = RegressionAnalysis(
            file_path, scale_data, polynomial_degree, model_type,
            missing_value_option, outlier_removal, scaling_option
        )
        reg.load_data()  # Load the data
        reg.preprocess_data()  # Preprocess the data
        
        # Display column statistics after preprocessing
        display_column_statistics(reg.X)
        
        if st.button("Run Regression Analysis"):
            results = reg.run_analysis()
            display_regression_results(results, model_type)

if __name__ == "__main__":
    main()
