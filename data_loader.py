# necessary libraries
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

#creation of the function to be called in main pipeline
def load_data():
    # Load the dataset
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection/SMSSpamCollection', sep='\t', names=['label', 'message'])
    
    # Convert labels to binary by mapping the target variable
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    #Split the dataset into training and test dataset
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

    #------------------ PREPROCESSING --------------------------------

    #loading of sklearn's vectorizer

    vectorizer = CountVectorizer()

    # employ vectorizer
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.fit_transform(X_test)

    return X_train_vectorized, X_test_vectorized, y_train, y_test

    
