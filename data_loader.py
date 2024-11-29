# Necessary libraries
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from preprocessing import preprocess_dataframe

# Creation of the function to be called in the main pipeline
def load_data():
    # Local folder of dataset
    path = './data/SMSSpamCollection.csv'
    
    # Load the dataset
    df = pd.read_csv(path, sep='\t', names=['label', 'message'])
    
    # Convert labels to binary by mapping the target variable
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Apply Preprocessing to the dataframe
    df = preprocess_dataframe(df)

    # Split the dataset into training and test dataset
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_message'], df['label'], test_size=0.2, random_state=42)

    #------------------ PREPROCESSING --------------------------------

    # Loading sklearn's vectorizer
    vectorizer = CountVectorizer()

    # Fit and transform the training data to avoid Data Leakage
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Transform the test data using the same vectorizer
    X_test_vectorized = vectorizer.transform(X_test)

    # ----------------- OVERSAMPLING CON SMOTE -----------------
    
    # Creates instance SMOTE
    smote = SMOTE(random_state=42)
    
    # Applies SMOTE over vectorized datasets in order to get the vectorial interpolations
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vectorized, y_train)
    
    # Get the datasets resampled (for training only)
    return X_train_resampled, X_test_vectorized, y_train_resampled, y_test, vectorizer