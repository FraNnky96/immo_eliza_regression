import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler

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
        self.X = self.df.drop(["Price"], axis=1)
        #self.X = self.df[['Living area','Bedrooms']].values
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
        # Apply scaling only to numeric columns
        # Select only numeric columns for scaling
        numeric_columns = self.X_train.select_dtypes(include=[np.number]).columns

        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()

        # Fit and transform the training data, and transform the test data
        #self.X_train = scaler.fit_transform(self.X_train)
        #self.X_test = scaler.transform(self.X_test)
        self.X_train[numeric_columns] = scaler.fit_transform(self.X_train[numeric_columns])
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
        # Initialize and apply CatBoostRegressor to the dataset.
        regressor = CatBoostRegressor(iterations=2500, depth=5, learning_rate=0.1, loss_function='RMSE', 
                                      logging_level='Verbose', train_dir='catboost_logs')
        
        # Specify categorical features by index (ensure these are correct)
        cat_features = [0,2,7,16,17,18]

        # Get training and testing data
        X_train, y_train = self.get_train_data()
        X_test, y_test = self.get_test_data()

        # Define eval_set (test set) for evaluation during training
        eval_set = [(X_test, y_test)]

        # Train the model and evaluate during training with the eval_set
        regressor.fit(X_train, y_train, verbose=100, cat_features=cat_features, eval_set=eval_set, plot=True)

        # After training, predict the test set results
        y_pred = regressor.predict(X_test)

        # Print predicted values
        print("Predicted Values:", y_pred)

        return y_test, y_pred

    def model_evaluation(self):
        # Apply model to get predictions
        y_test, y_pred = self.apply_model()

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Print the evaluation metrics
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R-squared: {r2*100:.2f}%")
        print(f"Mean Absolute Error: {mae:.2f}")
        

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

    # Apply the model and evaluate its performance
    data_format.model_evaluation()

if __name__ == "__main__":
    main()
