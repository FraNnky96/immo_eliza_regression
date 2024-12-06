from script.data_cleaning import *
from script.model import *
import time


def main():
    # Begin of time counting
    start_time = time.time()
    # Create an instance of the DataCleaner class
    data_cleaner = DataCleaner(r"data\data_set.csv")

    # Clean the data
    data_cleaner.clean_data()

    # Run the data checks and analysis
    data_cleaner.is_duplicate()
    data_cleaner.is_nan()
    data_cleaner.is_text()
    data_cleaner.corr()

    # Save the cleaned data to a new CSV
    data_cleaner.save_cleaned_data(r"data\clean_data.csv")

    # Load the cleaned data
    df = pd.read_csv(r"data\clean_data.csv")

    # Create an instance of the DataFormat class
    data_format = DataFormat(df)

    # Reshape the data (X and y)
    data_format.reshape()

    # Split the data into training and testing sets 80/20
    data_format.split()

    # Scale the data with RobustScaler
    data_format.scale_data()

    # Apply the model and get predictions test/train
    y_test, y_pred, y_train, y_pred_train = data_format.apply_model()

    # Evaluate the model and print metrics
    data_format.model_evaluation()

    # Visualize the model performance
    data_format.visualize_model_performance(y_test, y_pred, y_train, y_pred_train)

    # Save the trained model
    data_format.save_model()

    # Print the time it took to run
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")


# Running the main function
if __name__ == "__main__":
    main()
