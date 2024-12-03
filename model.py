import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

    def reshape(self):
        """
        Reshape the dataset to have X as features, y as target(price)
        """
        # X = features (drop 'Price')
        #self.X = self.df.drop(["Price"], axis=1)
        self.X = self.df[['Locality','Living area']].values
        # y = target (Price)
        self.y = self.df['Price'].values
        
        # Print y to verify it's reshaped correctly
        print("Shape of y:", self.y.shape)
        print("Shape of X:", self.X.shape)

    def split(self):
        """
        Splits the data into training and test sets.
        """
        # Splitting the dataset into the Training set and Test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)
        
        # Print the shapes of the split data
        print("Training set (features):", self.X_train.shape)
        print("Test set (features):", self.X_test.shape)
        print("Training set (target):", self.y_train.shape)
        print("Test set (target):", self.y_test.shape)

    def get_train_data(self):
        """
        return X_train, y_train
        """
        return self.X_train, self.y_train

    def get_test_data(self):
        """
        return X_test, y_test
        """
        return self.X_test, self.y_test
    
    def apply_model(self):
        """
        Initialize regressor with CatBoostRegressor
        Apply the model
        Predict the result
        """
        # Initialize the CatBoost Regressor
        regressor = CatBoostRegressor(iterations=500, depth=10, learning_rate=0.2, loss_function='RMSE')

        # Fitting CatBoost Regression to the Training set with eval_set to track performance on the test set
        X_train, y_train = self.get_train_data()
        X_test, y_test = self.get_test_data()

        # Define eval_set (test set) for evaluation during training
        eval_set = [(X_test, y_test)]

        # Train the model with eval_set to track performance on the test set
        regressor.fit(X_train, y_train, verbose=100, cat_features=[0], eval_set=eval_set)

        # After training, predict the test set results
        y_pred = regressor.predict(X_test)

        # Print predicted values
        print("Predicted Values:", y_pred)

        return y_test, y_pred

    
    def model_evaluation(self):
        """
        Evaluate the model
        """
        y_test, y_pred = self.apply_model()
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")


def main():
    # Load the data (assuming you already have the cleaned data loaded as df)
    df = pd.read_csv(r'data\clean_data.csv')  # Ensure this file path is correct
    
    # Create an instance of the DataFormat class
    data_format = DataFormat(df)
    
    # Reshape the data (X and y)
    data_format.reshape()
    
    # Split the data into training and testing sets
    data_format.split()

    # Apply the model
    # Evaluate the model
    data_format.model_evaluation()


# Running the main function
if __name__ == "__main__":
    main()

