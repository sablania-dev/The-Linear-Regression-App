import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import zscore
import os  # Ensure this import exists
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer  # MICE Imputer

class RegressionAnalysis:
    def __init__(self, file_path, scale_data=False, polynomial_degree=1, model_type='linear', handle_missing='None', outlier_removal=False, scaling_option="None"):
        self.file_path = file_path
        self.scale_data = scale_data
        self.polynomial_degree = polynomial_degree
        self.model_type = model_type
        self.handle_missing = handle_missing
        self.outlier_removal = outlier_removal
        self.scaling_option = scaling_option  # Store the scaling option
        self.data = None
        self.X = None
        self.y = None
        self.model = None
        self.results = {}
        self.images_dir = "images"
        os.makedirs(self.images_dir, exist_ok=True)  # Create the images folder if it doesn't exist
        
    def load_data(self):
        if self.file_path.endswith('.csv'):
            self.data = pd.read_csv(self.file_path)
        elif self.file_path.endswith(('.xls', '.xlsx')):
            self.data = pd.read_excel(self.file_path, engine='openpyxl')  # Specify the engine explicitly
        else:
            raise ValueError("Unsupported file format")
        
        # Handle missing values
        if self.handle_missing == "Drop Rows":
            self.data.dropna(inplace=True)
        elif self.handle_missing == "Fill with Mean":
            self.data.fillna(self.data.mean(), inplace=True)
        elif self.handle_missing == "Fill with Median":
            self.data.fillna(self.data.median(), inplace=True)
        elif self.handle_missing == "MICE Imputer":
            imputer = IterativeImputer(random_state=42)
            self.data.iloc[:, :] = imputer.fit_transform(self.data)
        elif self.handle_missing == "KNN Imputer":
            imputer = KNNImputer(n_neighbors=5)
            self.data.iloc[:, :] = imputer.fit_transform(self.data)
        
        # Outlier removal using Z-score
        if self.outlier_removal:
            self.data = self.data[(np.abs(zscore(self.data.select_dtypes(include=[np.number]))) < 3).all(axis=1)]
        
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1]
    
    def preprocess_data(self):
        self.X = pd.get_dummies(self.X, drop_first=True)  # Encode categorical variables
        if self.scale_data:
            if self.scaling_option == "Standard Scaling":
                scaler = StandardScaler()
            elif self.scaling_option == "Min-Max Scaling":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            self.X = scaler.fit_transform(self.X)
        
    def hyperparameter_tuning(self, X_train, y_train):
        param_grid = {}
        if self.model_type == 'ridge':
            param_grid = {'alpha': np.logspace(-3, 3, 10)}
            model = Ridge()
        elif self.model_type == 'lasso':
            param_grid = {'alpha': np.logspace(-3, 3, 10)}
            model = Lasso()
        else:
            return None
        
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(X_train, y_train)
        self.results['best_params'] = grid_search.best_params_
        
        # Save training vs validation loss plot
        plt.figure(figsize=(8, 6))
        plt.plot(param_grid['alpha'], -grid_search.cv_results_['mean_train_score'], label='Training Loss', marker='o')
        plt.plot(param_grid['alpha'], -grid_search.cv_results_['mean_test_score'], label='Validation Loss', marker='o')
        plt.xscale('log')
        plt.xlabel('Alpha')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training vs Validation Loss')
        plot_path = os.path.join(self.images_dir, "training_vs_validation_loss.png")
        plt.savefig(plot_path)  # Save the plot in images folder
        plt.close()  # Close the figure to free memory
        
        self.results['tuning_plot_path'] = plot_path  # Save the plot path in results
        return grid_search.best_estimator_
    
    def train_model(self):
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        if self.polynomial_degree > 1:
            poly = PolynomialFeatures(degree=self.polynomial_degree)
            X_train = poly.fit_transform(X_train)
            X_val = poly.transform(X_val)
            X_test = poly.transform(X_test)
        
        if self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type in ['ridge', 'lasso']:
            self.model = self.hyperparameter_tuning(X_train, y_train)
        else:
            raise ValueError("Unsupported model type")
        
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        y_test_pred = self.model.predict(X_test)
        
        # R² and Adjusted R² for training set
        r2_train = r2_score(y_train, y_train_pred)
        n_train, p_train = X_train.shape
        adj_r2_train = 1 - ((1 - r2_train) * (n_train - 1) / (n_train - p_train - 1))
        
        # R² and Adjusted R² for validation set
        r2_val = r2_score(y_val, y_val_pred)
        n_val, p_val = X_val.shape
        adj_r2_val = 1 - ((1 - r2_val) * (n_val - 1) / (n_val - p_val - 1))
        
        # R² and Adjusted R² for test set
        r2_test = r2_score(y_test, y_test_pred)
        n_test, p_test = X_test.shape
        adj_r2_test = 1 - ((1 - r2_test) * (n_test - 1) / (n_test - p_test - 1))
        
        # Store results for training set
        self.results['R2_train'] = r2_train
        self.results['Adjusted_R2_train'] = adj_r2_train
        self.results['MSE_train'] = mean_squared_error(y_train, y_train_pred)
        self.results['MAE_train'] = mean_absolute_error(y_train, y_train_pred)
        self.results['RMSE_train'] = np.sqrt(self.results['MSE_train'])
        
        # Store results for validation set
        self.results['R2_val'] = r2_val
        self.results['Adjusted_R2_val'] = adj_r2_val
        self.results['MSE_val'] = mean_squared_error(y_val, y_val_pred)
        self.results['MAE_val'] = mean_absolute_error(y_val, y_val_pred)
        self.results['RMSE_val'] = np.sqrt(self.results['MSE_val'])
        
        # Store results for test set
        self.results['R2_test'] = r2_test
        self.results['Adjusted_R2_test'] = adj_r2_test
        self.results['MSE_test'] = mean_squared_error(y_test, y_test_pred)
        self.results['MAE_test'] = mean_absolute_error(y_test, y_test_pred)
        self.results['RMSE_test'] = np.sqrt(self.results['MSE_test'])
        
        if self.model_type == 'linear':
            X_train_const = sm.add_constant(X_train)
            ols_model = sm.OLS(y_train, X_train_const).fit()
            self.results['coefficients'] = ols_model.params
            self.results['std_errors'] = ols_model.bse
            self.results['p_values'] = ols_model.pvalues
    
    def plot_residuals(self):
        y_pred = self.model.predict(self.X)
        residuals = self.y - y_pred
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, bins=30, kde=True)
        plt.title('Residuals Distribution')
        plt.savefig(os.path.join(self.images_dir, "residuals_plot.png"))  # Save the plot in images folder
        plt.close()  # Close the figure to free memory
    
    def run_analysis(self):
        self.load_data()
        self.preprocess_data()
        self.train_model()
        # Removed call to plot_distributions
        self.plot_residuals()
        return self.results  # Return the results instead of printing

# Example Usage
# reg = RegressionAnalysis('medical_costs.csv', scale_data=True, polynomial_degree=2, model_type='ridge', handle_missing='Fill with Mean', outlier_removal=True)
# reg.run_analysis()
