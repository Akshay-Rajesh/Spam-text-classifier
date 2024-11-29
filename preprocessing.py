import re
from nltk.corpus import stopwords

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

def preprocess_dataframe(df):
    """
    Preprocesses the input DataFrame:
    - Cleans the text messages
    - Adds additional features
    """
    df['cleaned_message'] = df['message'].apply(clean_text)
    # df = add_features(df)
    return df
