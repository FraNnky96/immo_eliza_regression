import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class DataCleaner:
    def __init__(self, file_path):
        """
        Initialize the DataCleaner class and load the dataset.

        Parameters:
        file_path (str): Path to the CSV file containing the dataset.
        """
        try:
            self.df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            raise
        except pd.errors.EmptyDataError:
            print(f"Error: File '{file_path}' is empty.")
            raise
        except Exception as e:
            print(f"An error occurred while loading the file: {e}")
            raise

    def clean_data(self):
        """
        Cleans the dataset by handling missing values, encoding categorical variables, and removing outliers.
        """
        # Drop rows with NaN in important columns
        self.df = self.df.dropna(subset=["Bedrooms", "Living area"])

        # Fill missing values with mode (for categorical) and median (for numerical)
        self.df["Facades"] = self.df["Facades"].fillna(self.df["Facades"].mode()[0])
        self.df["Surface of the plot"] = self.df["Surface of the plot"].fillna(
            self.df["Surface of the plot"].median()
        )

        # Mapping dictionary for locality replacement
        locality_mapping = {
            "Brugge": "Bruges",
            "Anvers": "Antwerp",
            "Courtrai": "Kortrijk",
            "Gand": "Gent",
            "Bruxelles": "Brussels",
            "Malines": "Mechelen",
            "Louvain": "Leuven",
            "Ostende": "Oostend",
            "Saint-Nicolas": "Sint-Niklaas",
            "Hal-Vilvorde": "Halle-Vilvoorde",
            "Audenarde": "Oudenaarde",
            "Dendermonde": "Termonde",
            "Alost": "Aalst",
            "Roulers": "Roeselare",
            "Ypres": "Ieper",
            "Tongres": "Tongeren",
            "Dixmude": "Diksmuide",
            "Veurne": "Furnes",
        }

        # Replace locality names using the mapping
        self.df["Locality"] = self.df["Locality"].replace(locality_mapping)

        # Remove rows with certain property type and outliers based on price
        self.df = self.df[self.df["Property type"] != "Apartment_Block"]
        self.df = self.df[self.df["Price"] <= 7500000]

    def is_duplicate(self):
        """
        Check if there are any duplicate rows in the dataset.
        """
        if self.df.duplicated().any():
            print("There are duplicates in the dataset.")
        else:
            print("No duplicates found.")

    def is_nan(self):
        """
        Check for NaN values in the dataset and print details of any missing values.
        """
        nan_summary = self.df.isna().sum()
        nan_columns = nan_summary[nan_summary > 0]
        if nan_columns.empty:
            print("No NaN values in the dataset.")
        else:
            print("Columns with NaN values:")
            for column, count in nan_columns.items():
                print(f"{column}: {count} missing values")

    def is_text(self):
        """
        Count the number of columns that contain text (string) values.
        """
        count = (self.df.dtypes == "object").sum()
        print(f"{count} columns contain text values.")

    def corr(self):
        """
        Display and plot the correlation matrix for numerical columns.
        """
        numerical_cols = self.df[
            [
                "Zip code",
                "Price",
                "Bedrooms",
                "Living area",
                "Surface of the plot",
                "Facades",
                "Fireplace",
                "Garden",
                "Garden surface",
                "Terrace",
                "Terrace surface",
                "Furnished",
                "Swimming pool",
            ]
        ]
        # Calculate and plot the correlation matrix
        corr_matrix = numerical_cols.corr()

        # Plotting the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.show()

    def save_cleaned_data(self, output_path):
        """
        Save the cleaned dataset to a CSV file.

        Parameters:
        output_path (str): The file path where the cleaned data will be saved.
        """
        try:
            self.df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to {output_path}")
        except Exception as e:
            print(f"An error occurred while saving the cleaned data: {e}")
