import pandas as pd
from pandas_profiling import ProfileReport

# Load the dataset
df = pd.read_csv('clean_data.csv')

# Generate the profiling report
profile = ProfileReport(df, title="Dataset Profile Report", explorative=True)

# Save the report as an HTML file
profile.to_file("data_profile_report.html")

# (Optional) Display in Jupyter Notebook
profile.to_notebook_iframe()
