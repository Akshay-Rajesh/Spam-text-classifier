import pandas as pd
import joblib  # For model/vectorizer loading, consider using joblib or pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_vectorizer(vectorizer_path):
    vectorizer = joblib.load(vectorizer_path)
    return vectorizer

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def load_preprocessed_data(data_path):
    df = pd.read_csv(data_path)
    return df['cleaned_message'], df['LABEL']


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    metrics = {
        'Accuracy': accuracy_score(y, predictions),
        'Precision': precision_score(y, predictions, average='macro'),
        'Recall': recall_score(y, predictions, average='macro'),
        'F1 Score': f1_score(y, predictions, average='macro')
    }
    # Display and save the confusion matrix
    conf_matrix = confusion_matrix(y, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    return metrics


def main():
    vectorizer_path = './vectorizer.pkl'
    model_path = './model.pkl'
    data_path = './data/Preprocessed_Dataset_5971.csv'
    
    # Load the preprocessed data
    messages, labels = load_preprocessed_data(data_path)
    vectorizer = load_vectorizer(vectorizer_path)
    
    # Transform the text data to feature vectors
    X = vectorizer.transform(messages)  # Ensure 'cleaned_message' matches the column used for text

    # Load the model
    model = load_model(model_path)
    
    # Evaluate the model
    metrics = evaluate_model(model, X, labels)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main()
