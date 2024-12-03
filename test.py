import pandas as pd

df = pd.read_csv('data\clean_data.csv')

for column in df.columns:
    print(f"Unique values in column '{column}':")
    print(df[column].unique())
    print()