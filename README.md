# The Linear Regression App

## Introduction

This project is a regression analysis tool that allows users to perform different types of regression (Linear, Polynomial, Ridge, Lasso) on datasets provided as CSV or Excel files. The tool provides insights such as regression coefficients, standard errors, significance tests, and ANOVA tables. Additionally, it offers hyperparameter tuning using GridSearchCV, data visualization, and feature preprocessing options.

## Features

- Supports CSV and Excel file formats.
- Handles missing values (drop rows, fill with mean/median).
- Detects and removes outliers using Z-score.
- Allows feature scaling using StandardScaler.
- Supports polynomial regression (user-defined degree).
- Implements Linear, Ridge, and Lasso regression models.
- Performs hyperparameter tuning using GridSearchCV.
- Provides regression coefficients, standard errors, and p-values.
- Generates ANOVA tables for statistical analysis.
- Displays various plots:
  - Feature distributions
  - Residual plots
  - Training loss vs validation loss (for hyperparameter tuning)

## Installation

Clone the repository:

```bash
git clone https://github.com/sablania-dev/The-Linear-Regression-App
```

Navigate to the project directory:

```bash
cd regression-analysis
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the backend server:

```bash
python regression_backend.py
```

Run the Streamlit frontend:

```bash
streamlit run regression_frontend.py
```

Upload a CSV or Excel file and select the desired regression settings through the Streamlit UI.

## How It Works

### Data Preprocessing:

- Loads the dataset and handles missing values.
- Detects and removes outliers using Z-score (optional).
- Encodes categorical features using one-hot encoding.
- Scales numerical features if selected.
- All columns except the last will be considered as input features, and the last column will be the output target.

### Model Training & Evaluation:

- Splits the dataset into training and testing sets.
- Trains the selected regression model (Linear, Ridge, or Lasso).
- If using Ridge or Lasso, performs hyperparameter tuning with GridSearchCV.
- Computes regression metrics: R², MSE, MAE, RMSE.

### Statistical Analysis & Visualization:

- Computes and displays regression coefficients, standard errors, and p-values.
- Generates an ANOVA table for significance testing.
- Displays feature distributions, residual plots, and hyperparameter tuning graphs.
