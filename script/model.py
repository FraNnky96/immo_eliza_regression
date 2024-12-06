import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
)
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import shap


class DataFormat:
    def __init__(self, df):
        """
        Initialize the class DataFormat and inport the dataset as Dataframe

        :param df: Dataset
        """
        self.df = df
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.regressor = None

    def reshape(self):
        """
        Reshape the dataset to have X as features, y as target(price)
        """
        # X = features (drop 'Price')
        self.X = self.df.drop(["Price"], axis=1)
        self.y = self.df["Price"].values

        # Ensure categorical features are treated correctly (as strings)
        self.X["Building condition"] = self.X["Building condition"].astype(str)
        self.X["Property"] = self.X["Property"].astype(str)

        # Print shapes of X and y to ensure correct reshaping
        print("Shape of y:", self.y.shape)
        print("Shape of X:", self.X.shape)

    def split(self):
        """
        Splits the data into training and test sets 80/20 with a random state of 0.
        """
        # Split the dataset into training and test sets (80% training, 20% testing)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=0
        )

    def scale_data(self):
        """
        Scale the numericals features with RobustScaler
        """
        # Select only numeric columns for scaling
        numeric_columns = self.X_train.select_dtypes(include=[np.number]).columns

        # Initialize the RobustScaler
        scaler = RobustScaler()

        # Fit and transform the training data, and transform the test data
        self.X_train[numeric_columns] = scaler.fit_transform(
            self.X_train[numeric_columns]
        )
        self.X_test[numeric_columns] = scaler.transform(self.X_test[numeric_columns])

        print("Data scaled successfully.")

    def get_train_data(self):
        """
        Return training data: X_train, y_train
        """
        return self.X_train, self.y_train

    def get_test_data(self):
        """
        Return testing data: X_test, y_test
        """
        return self.X_test, self.y_test

    def apply_model(self):
        """
        Define hyperparameter for Grid search
        Instantiate the Catboost regressor
        Fit the model
        Get the best model and perform cross validation with Kfold
        Display informations about metrics (R*2 score,positive MSE)
        Predict the value on the train and test set of data
        """
        # Hyperparameter grid for GridSearchCV
        param_grid = {
            "iterations": [500, 1000],
            "depth": [6, 8],
            "learning_rate": [0.1, 0.2],
            "early_stopping_rounds": [50],
        }
        cat_features = [
            "Locality",
            "Property type",
            "Building condition",
            "Province",
            "Property",
            "Region",
        ]

        # Create a CatBoostRegressor object
        regressor = CatBoostRegressor(loss_function="RMSE", cat_features=cat_features)

        # Perform GridSearchCV
        grid_search = GridSearchCV(
            estimator=regressor, param_grid=param_grid, cv=2, n_jobs=-1, verbose=1
        )

        # Fit the model with the best parameters
        grid_search.fit(self.X_train, self.y_train)
        print("Best parameters:", grid_search.best_params_)

        # Get the best model from GridSearchCV
        self.regressor = grid_search.best_estimator_

        # Cross-validation: perform KFold cross-validation
        kfold = KFold(
            n_splits=5, shuffle=True, random_state=0
        )  # 5-fold cross-validation

        # Perform cross-validation and compute the R^2 score
        cv_results = cross_val_score(
            self.regressor,
            self.X_train,
            self.y_train,
            cv=kfold,
            scoring="neg_mean_squared_error",
        )

        # The result is negative MSE, so we will convert it to positive MSE
        print(f"Cross-validation results (Negative MSE): {cv_results}")
        print(f"Mean Negative MSE: {np.mean(cv_results)}")
        print(f"Standard Deviation of Negative MSE: {np.std(cv_results)}")

        # Predict on the test and train sets and print it
        y_pred = self.regressor.predict(self.X_test)
        y_pred_train = self.regressor.predict(self.X_train)

        print("Predicted Test Values:", y_pred)
        print("Predicted Train Values:", y_pred_train)

        return self.y_test, y_pred, self.y_train, y_pred_train

    def plot_prediction_combined(self, y_train, y_pred_train, y_test, y_pred_test):
        """
        Plot with a scatter plot the prediction on test and train with a perfect line for the prediction

        :param y_train: Value y for train
        :param y_pred_train: Predicted value of y on train
        :y_test: Value of y on test
        :param y_pred_test: Predicted value of y on test
        """
        plt.figure(figsize=(8, 6))

        # Plot for Train Data
        plt.scatter(
            y_train,
            y_pred_train,
            alpha=0.6,
            color="orange",
            label="Train Data",
            marker="o",
        )

        # Plot for Test Data
        plt.scatter(
            y_test, y_pred_test, alpha=0.6, color="blue", label="Test Data", marker="x"
        )

        # Line for Perfect Prediction
        plt.plot(
            [min(min(y_train), min(y_test)), max(max(y_train), max(y_test))],
            [min(min(y_train), min(y_test)), max(max(y_train), max(y_test))],
            color="red",
            linestyle="--",
            label="Perfect Prediction",
        )

        # Labels and Title
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("Prediction vs Actual (Train and Test Sets)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_residuals_combined(self, y_train, y_pred_train, y_test, y_pred_test):
        """
        Plot the residuals of test and train of the prediction with a perfect line of prediction

        :param y_train: Value y for train
        :param y_pred_train: Predicted value of y on train
        :y_test: Value of y on test
        :param y_pred_test: Predicted value of y on test
        """
        residuals_train = y_train - y_pred_train
        residuals_test = y_test - y_pred_test

        plt.figure(figsize=(8, 6))

        # Plot residuals for Train Data
        plt.scatter(
            y_train,
            residuals_train,
            alpha=0.6,
            color="orange",
            label="Train Residuals",
            marker="o",
        )

        # Plot residuals for Test Data
        plt.scatter(
            y_test,
            residuals_test,
            alpha=0.6,
            color="blue",
            label="Test Residuals",
            marker="x",
        )

        # Line for Zero Residual
        plt.axhline(0, color="red", linestyle="--", label="Zero Residual Line")

        # Labels and Title
        plt.xlabel("True Values")
        plt.ylabel("Residuals")
        plt.title("Residuals (Train and Test Sets)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_feature_importance(self, model, X_columns):
        """
        Plots the feature importance using SHAP.

        :param model: The trained model (CatBoostRegressor)
        :param X_columns: The feature columns of the dataset.
        """
        # Create SHAP explainer
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(self.X_test)

        # Print feature names to ensure they exist
        print("Features in the dataset:", X_columns)

        # Create SHAP summary plot to show feature importance
        shap.summary_plot(shap_values, self.X_test, plot_type="bar")

    def visualize_model_performance(self, y_test, y_pred_test, y_train, y_pred_train):
        """
        Visualizes the model performance with predictions, residuals, and feature importance.

        :param y_train: Value y for train
        :param y_pred_train: Predicted value of y on train
        :y_test: Value of y on test
        :param y_pred_test: Predicted value of y on test
        """
        # Prediction vs Actual (combined)
        self.plot_prediction_combined(y_train, y_pred_train, y_test, y_pred_test)

        # Residual Plot (combined)
        self.plot_residuals_combined(y_train, y_pred_train, y_test, y_pred_test)

        # Feature Importance
        self.plot_feature_importance(self.regressor, self.X.columns)

    def save_model(self):
        """
        Save the model after training
        """
        if self.regressor:
            self.regressor.save_model("model.cbm")
            print("Model saved as 'model.cbm'")
        else:
            print("Model has not been trained yet. Please train the model first.")

    def model_evaluation(self):
        """
        Evaluates the performance of the model on test and train datasets
        using several regression metrics: Mean Absolute Error (MAE),
        Mean Squared Error (MSE), R-squared (R²), Explained Variance Score (EVS), Mean absolute percentage error (MAPE) and Symmetric mean absolute percentage error (sMAPE).

        Display all the results
        """
        # Apply the model and get predictions for both training and test datasets
        y_test, y_pred, y_train, y_pred_train = self.apply_model()

        # Evaluation metrics
        mae_test = mean_absolute_error(y_test, y_pred)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred)
        mse_train = mean_squared_error(y_train, y_pred_train)
        rmse_test = np.sqrt(mse_test)
        rmse_train = np.sqrt(mse_train)
        r2_test = r2_score(y_test, y_pred)
        r2_train = r2_score(y_train, y_pred_train)
        evs_test = explained_variance_score(y_test, y_pred)
        evs_train = explained_variance_score(y_train, y_pred_train)
        mape_test = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        mape_train = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
        smape_test = 100 * np.mean(
            2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred))
        )
        smape_train = 100 * np.mean(
            2
            * np.abs(y_train - y_pred_train)
            / (np.abs(y_train) + np.abs(y_pred_train))
        )

        # Print evaluation metrics
        print(f"Mean Absolute Error (test): {mae_test:.2f}")
        print(f"Mean Absolute Error (train): {mae_train:.2f}")
        print(f"Root Mean Squared Error (test): {rmse_test:.2f}")
        print(f"Root Mean Squared Error (train): {rmse_train:.2f}")
        print(f"R-squared (test): {r2_test*100:.2f}%")
        print(f"R-squared (train): {r2_train*100:.2f}%")
        print(f"Mean Squared Error (test): {mse_test:.2f}")
        print(f"Mean Squared Error (train): {mse_train:.2f}")
        print(f"Explained Variance Score (test): {evs_test:.2f}")
        print(f"Explained Variance Score (train): {evs_train:.2f}")
        print(f"Mean Absolute Percentage Error (test): {mape_test:.2f}%")
        print(f"Mean Absolute Percentage Error (train): {mape_train:.2f}%")
        print(f"Symmetric Mean Absolute Percentage Error (test): {smape_test:.2f}%")
        print(f"Symmetric Mean Absolute Percentage Error (train): {smape_train:.2f}%")
