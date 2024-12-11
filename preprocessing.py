import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE

def clean_text(message):
    """
    Cleans the input message by:
    - Removing specific characters (like '.')
    - Keeping numbers and special characters like '$', '%', and '!'
    - Removing stopwords
    - Normalizing spaces
    """
    # Load stopwords
    stop = set(stopwords.words('english'))
    
    # Remove dots but keep special characters and numbers
    message = re.sub(r'[.,;]', '', message)
    
    # Remove stopwords
    message = ' '.join(word for word in message.split() if word.lower() not in stop)
    
    # Normalize spaces
    message = re.sub(r'\s+', ' ', message).strip()
    
    return message

def vectorize_data(X_train, X_test):
    """
    Vectorizes the training and test datasets using CountVectorizer.
    Ensures that the vectorizer is fitted only with the training data.
    """
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    return X_train_vectorized, X_test_vectorized, vectorizer

def apply_smote(X_train, y_train):
    """
    Applies SMOTE to address class imbalance issues in the training set.
    """
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def preprocess_dataframe(df):
    """
    Preprocesses the input DataFrame:
    - Cleans the text messages
    - Adds additional features
    """
    df['cleaned_message'] = df['message'].apply(clean_text)
    # df = add_features(df)
    return df
