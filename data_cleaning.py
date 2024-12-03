import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class DataCleaner:
    def __init__(self, file_path):
        # Initialize with the file path, and load the data
        self.df = pd.read_csv(file_path)

    def clean_data(self):
        """
        Cleans the dataset by handling missing values and encoding categorical variables.
        """
        # Cleaning for NaN values
        self.df = self.df.dropna(subset=["Bedrooms", "Living area"])

        # Filling missing values for specific columns with median
        self.df["Facades"] = self.df["Facades"].fillna(self.df["Facades"].mode()[0])
        self.df["Surface of the plot"] = self.df["Surface of the plot"].fillna(self.df["Surface of the plot"].median())

        # joining flemish name with french name in locality
        self.df.loc[self.df["Locality"] == "Brugge", "Locality"] = "Bruges"
        self.df.loc[self.df["Locality"] == "Anvers", "Locality"] = "Antwerp"
        self.df.loc[self.df["Locality"] == "Courtrai", "Locality"] = "Kortrijk"
        self.df.loc[self.df["Locality"] == "Gand", "Locality"] = "Gent"
        self.df.loc[self.df["Locality"] == "Bruxelles", "Locality"] = "Brussels"
        self.df.loc[self.df["Locality"] == "Malines", "Locality"] = "Mechelen"
        self.df.loc[self.df["Locality"] == "Louvain", "Locality"] = "Leuven"
        self.df.loc[self.df["Locality"] == "Ostende", "Locality"] = "Oostend"
        self.df.loc[self.df["Locality"] == "Saint-Nicolas", "Locality"] = "Sint-Niklaas"
        self.df.loc[self.df["Locality"] == "Hal-Vilvorde", "Locality"] = "Halle-Vilvoorde"
        self.df.loc[self.df["Locality"] == "Audenarde", "Locality"] = "Oudenaarde"
        self.df.loc[self.df["Locality"] == "Dendermonde", "Locality"] = "Termonde"
        self.df.loc[self.df["Locality"] == "Alost", "Locality"] = "Aalst"
        self.df.loc[self.df["Locality"] == "Roulers", "Locality"] = "Roeselare"
        self.df.loc[self.df["Locality"] == "Ypres", "Locality"] = "Ieper"
        self.df.loc[self.df["Locality"] == "Tongres", "Locality"] = "Tongeren"
        self.df.loc[self.df["Locality"] == "Dixmude", "Locality"] = "Diksmuide"
        self.df.loc[self.df["Locality"] == "Veurne", "Locality"] = "Furnes"

        # dropping some rows

        self.df = self.df[self.df['Property type'] != "Apartement_Block"]
        


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
        Check for NaN values in the dataset and display detailed information about them.
        """
        nan_summary = self.df.isna().sum()
        nan_columns = nan_summary[nan_summary > 0]  # Only show columns with NaN values
        if nan_columns.empty:
            print("No NaN values in the dataset.")
        else:
            print("Columns with NaN values:")
            for column, count in nan_columns.items():
                print(f"{column}: {count} missing values")

    def is_text(self):
        """
        Count how many columns have text values.
        """
        count = (self.df.dtypes == "object").sum()
        print(f"{count} columns have text values")

    def corr(self):
        """
        Show correlation matrix of numerical columns.
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
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.show()

    def save_cleaned_data(self, output_path):
        """
        Save the cleaned data to a CSV file.
        """
        self.df.to_csv(output_path, index=False)


# Main execution of the class


def main():
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


# Running the main function
if __name__ == "__main__":
    main()
