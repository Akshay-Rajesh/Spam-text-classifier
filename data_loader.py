# Necessary libraries
import pandas as pd 
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_dataframe, vectorize_data, apply_smote

# Creation of the function to be called in the main pipeline
def load_data():
    """
    Loads, preprocesses, and splits the SMS spam collection dataset.
    Returns the processed training and test sets along with the vectorizer and resampled labels.
    """
    # Local folder of dataset
    path = './data/SMSSpamCollection.csv'
    
    # Load the dataset
    df = pd.read_csv(path, sep='\t', names=['label', 'message'])
    
    # Convert labels to binary by mapping the target variable
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Apply Preprocessing to the dataframe
    df = preprocess_dataframe(df, "message")

    # Split the dataset into training and test dataset
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_message'], df['label'], test_size=0.3, random_state=42)

    #------------------ VECTORIZATION --------------------------------

    # Vectorize the training and test datasets
    X_train_vectorized, X_test_vectorized, vectorizer = vectorize_data(X_train, X_test)

    # ----------------- OVERSAMPLING WITH SMOTE -----------------
    
    # Apply SMOTE to the training dataset
    X_train_resampled, y_train_resampled = apply_smote(X_train_vectorized, y_train)
 
    # Get the datasets resampled (for training only)
    return X_train_resampled, X_test_vectorized, y_train_resampled, y_test, vectorizer