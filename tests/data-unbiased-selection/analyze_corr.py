# one off test script used for debugging unbiased selection
import pdb
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# List of file names
file_names = [f'data-indep{i}.txt' for i in range(0,1000)]

# Initialize an empty DataFrame
result_df = pd.DataFrame(columns=['File', 'MaxCorrelationColumn'])

# Loop over each file
for file_name in file_names:
    # Read data into Pandas DataFrame
    df = pd.read_csv(file_name, delim_whitespace=True)

    # Assuming 'Y' is the column you are interested in
    # Change it accordingly if the actual column name is different
    y_column = 'Y'

    # Calculate Pearson correlation coefficient for each column with 'Y'
    correlations = {col: np.abs(pearsonr(df[y_column], df[col])[0]) for col in df.columns if col != y_column}

    # Find the column with the maximum correlation
    max_corr_column = max(correlations, key=correlations.get)
    # Append the result to the result DataFrame
    result_df = pd.concat([result_df, pd.DataFrame({'File': [file_name], 'MaxCorrelationColumn': [max_corr_column]})], ignore_index=True)

# Display the result
print(result_df.MaxCorrelationColumn.value_counts())

