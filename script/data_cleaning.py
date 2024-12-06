import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class DataCleaner:
    def __init__(self, file_path):
        """
        Initialize the DataCleaner Class
        Load the data set with the file path
        """
        self.df = pd.read_csv(file_path)

    def clean_data(self):
        """
        Cleans the dataset by handling missing values and encoding categorical variables.
        """
        # Cleaning for NaN values
        self.df = self.df.dropna(subset=["Bedrooms", "Living area"])

        # Filling missing values for specific columns with median
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

        # dropping some rows

        self.df = self.df[self.df["Property type"] != "Apartment_Block"]
        outliers = self.df.loc[self.df["Price"] > 7500000]
        self.df = self.df.drop(outliers.index)

    def is_duplicate(self):
        """
        Check if there are any duplicates in the data.
        """
        if self.df.duplicated().any():
            print("There are duplicates")
        else:
            print("No duplicates")

    def is_nan(self):
        """
        Check for NaN values in the dataset and display detailed information about them if there's some.
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
        Count how many columns have text values and return it.
        """
        count = (self.df.dtypes == "object").sum()
        print(f"{count} columns have text values")

    def corr(self):
        """
        Show correlation matrix of the numericals columns.
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
        # Calculating correlation matrix
        corr_matrix = numerical_cols.corr()

        # Plotting the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.show()

    def save_cleaned_data(self, output_path):
        """
        Save the cleaned data to a CSV file.
        """
        self.df.to_csv(output_path, index=False)
