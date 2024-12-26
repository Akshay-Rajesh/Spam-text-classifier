import pandas as pd
from preprocessing import preprocess_dataframe  # Assuming this function is defined in your preprocessing.py
import nltk

nltk.download('stopwords')

def load_and_merge_datasets(new_data_path, old_data_path):

    # Load the new and old datasets
    new_df = pd.read_csv(new_data_path)
    old_df = pd.read_csv(old_data_path, sep='\t', names=['label', 'message'])
    
    # Check for duplicates by merging new data with old data based on 'TEXT' column
    combined_df = pd.concat([new_df, old_df], ignore_index=True)
    unique_new_data = combined_df.drop_duplicates(subset=['TEXT'], keep=False)

    # Filtering out the old data entries
    unique_new_data = unique_new_data[~unique_new_data['TEXT'].isin(old_df['message'])]

    # Filter for 'ham' and 'spam' labels only
    filtered_data = unique_new_data[unique_new_data['LABEL'].isin(['ham', 'spam'])]
    filtered_data['LABEL'] = filtered_data['LABEL'].map({'ham': 0, 'spam': 1})

    return filtered_data

def preprocess_data(df):

    # Specify the correct column name for text data
    preprocessed_df = preprocess_dataframe(df, text_column_name='TEXT')
    return preprocessed_df

def load_model_from_mlflow(run_id):
    
    model_uri = f"mlruns:/{run_id}/model"  # Asegúrate de que el nombre del artefacto sea correcto, aquí se usa 'model'
    model = mlflow.pyfunc.load_model(model_uri)
    return model

def main():
    # Paths to the datasets
    new_data_path = './data/Dataset_5971.csv'
    old_data_path = './data/SMSSpamCollection.csv'
    
    # Load and merge datasets
    new_data_clean = load_and_merge_datasets(new_data_path, old_data_path)
    
    # Preprocess the data
    new_data_preprocessed = preprocess_data(new_data_clean)
    
    # Optionally, save the preprocessed new data to a new CSV file
    new_data_preprocessed.to_csv('./data/Preprocessed_Dataset_5971.csv', index=False)

if __name__ == '__main__':
    main()