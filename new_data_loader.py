import pandas as pd
from preprocessing import preprocess_dataframe  # Assuming this function is defined in your preprocessing.py

def load_and_merge_datasets(new_data_path, old_data_path):
    """
    Load the new and old datasets, and remove any duplicates.
    Args:
        new_data_path (str): Path to the new dataset CSV file.
        old_data_path (str): Path to the existing dataset CSV file.
    Returns:
        pd.DataFrame: A DataFrame of the new data with duplicates removed.
    """
    # Load the new and old datasets
    new_df = pd.read_csv(new_data_path)
    old_df = pd.read_csv(old_data_path)
    
    # Concatenate both DataFrames
    combined_df = pd.concat([new_df, old_df]).drop_duplicates(subset=['TEXT'], keep=False)
    
    # Return only the new data entries
    unique_new_data = combined_df[~combined_df['TEXT'].isin(old_df['TEXT'])]
    return unique_new_data