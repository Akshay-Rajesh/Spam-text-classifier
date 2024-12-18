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

def preprocess_data(df):
    """
    Apply preprocessing to the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing the new data.
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Specify the correct column name for text data
    preprocessed_df = preprocess_dataframe(df, text_column_name='TEXT')
    return preprocessed_df

def main():
    # Paths to the datasets
    new_data_path = './data/new/Dataset_5971.csv'
    old_data_path = './data/SMSSpamCollection.csv'
    
    # Load and merge datasets
    new_data_clean = load_and_merge_datasets(new_data_path, old_data_path)
    
    # Preprocess the data
    new_data_preprocessed = preprocess_data(new_data_clean)
    
    # Optionally, save the preprocessed new data to a new CSV file
    new_data_preprocessed.to_csv('./data/new/Preprocessed_Dataset_5971.csv', index=False)

if __name__ == '__main__':
    main()