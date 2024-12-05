import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

class DataFormat:
    def __init__(self, df):
        """
        Initialize the class
        """
        # Initialize with the DataFrame
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
        self.X = self.df.drop(["Price",], axis=1)
        # y = target (Price)
        self.y = self.df['Price'].values
        
        # Print shapes of X and y to ensure correct reshaping
        print("Shape of y:", self.y.shape)
        print("Shape of X:", self.X.shape)

    def split(self):
        """
        Splits the data into training and test sets.
        """
        # Split the dataset into training and test sets (80% training, 20% testing)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)

    def scale_data(self):
        # Select only numeric columns for scaling
        numeric_columns = self.X_train.select_dtypes(include=[np.number]).columns

        # Initialize the MinMaxScaler
        scaler = RobustScaler()

        # Fit and transform the training data, and transform the test data
        self.X_train[numeric_columns] = scaler.fit_transform(self.X_train[numeric_columns].copy())
        self.X_test[numeric_columns] = scaler.transform(self.X_test[numeric_columns].copy())

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
        # Get training and testing data
        X_train, y_train = self.get_train_data()
        X_test, y_test = self.get_test_data()

        # Set hyperparameter grid for GridSearchCV
        param_grid = {
            'iterations': [1000],
            'depth': [6],
            'learning_rate': [0.1],
            'early_stopping_rounds': [50]
        }

        cat_features = ['Locality', 'Property type','Building condition','Province','Property','Region']

        # Create a CatBoostRegressor object
        regressor = CatBoostRegressor(loss_function='RMSE', cat_features=cat_features)

        # Perform GridSearchCV
        grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=2, n_jobs=-1, verbose=1)

        # Fit the model with the best parameters
        grid_search.fit(X_train, y_train)
        print("Best parameters:", grid_search.best_params_)

        # Get the best model from GridSearchCV and evaluate it
        self.regressor = grid_search.best_estimator_

        # Evaluate the model with the test set
        y_pred = self.regressor.predict(X_test)
        
        print("Predicted Values:", y_pred)

        return y_test, y_pred
    
    def plot_prediction_vs_actual(self, y_test, y_pred):
        plt.figure(figsize=(8,6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Line of perfect prediction
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("Prediction vs Actual")
        plt.show()

    def plot_residuals(self, y_test, y_pred):
        residuals = y_test - y_pred
        plt.figure(figsize=(8,6))
        plt.scatter(y_test, residuals, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')  # Line at zero for reference
        plt.xlabel("True Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.show()

    def plot_feature_importance(self, model, feature_names):
        feature_importances = model.get_feature_importance()
        
        # Create a dataframe of feature importances
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10,6))
        plt.barh(feature_df['Feature'], feature_df['Importance'], color='skyblue')
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Plot')
        plt.show()

    def visualize_model_performance(self, y_test, y_pred):
        # 1. Prediction vs Actual
        self.plot_prediction_vs_actual(y_test, y_pred)
        
        # 2. Residual Plot
        self.plot_residuals(y_test, y_pred)
        
        # 3. Feature Importance Plot
        self.plot_feature_importance(self.regressor, self.X.columns)

    def save_model(self):
        """
        Save the trained CatBoost model to a file.
        """
        if self.regressor:
            self.regressor.save_model('model.cbm')
            print("Model saved as 'model.cbm'")
        else:
            print("Model has not been trained yet. Please train the model first.")

    def model_evaluation(self):
        # Apply model to get predictions
        y_test, y_pred = self.apply_model()

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
        
        # Print the evaluation metrics
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R-squared: {r2*100:.2f}%")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Explained Variance Score: {evs:.2f}")   

        

def main():
    # Load the cleaned data
    df = pd.read_csv(r'data\clean_data.csv')  # Ensure this file path is correct
    
    # Create an instance of the DataFormat class
    data_format = DataFormat(df)
    
    # Reshape the data (X and y)
    data_format.reshape()
    
    # Split the data into training and testing sets
    data_format.split()

    # Scale the data
    data_format.scale_data()

    # Apply the model and get predictions
    y_test, y_pred = data_format.apply_model()

    # Evaluate the model
    data_format.model_evaluation()

    # Visualize the model performance
    data_format.visualize_model_performance(y_test, y_pred)

    # Save the trained model
    data_format.save_model()

if __name__ == "__main__":
    main()



