import pandas as pd
from sklearn.externals import joblib  # For model loading, consider using joblib or pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_preprocessed_data(file_path):
    """
    Load the preprocessed dataset from a CSV file.
    Args:
        file_path (str): Path to the CSV file containing the preprocessed data.
    Returns:
        X (pd.DataFrame): Features ready for model prediction.
        y (pd.Series): True labels.
    """
    df = pd.read_csv(file_path)
    X = df.drop('LABEL', axis=1)  # Assuming 'LABEL' is the target column
    y = df['LABEL']
    return X, y

def load_model(model_path):
    """
    Load a trained model from a file.
    Args:
        model_path (str): Path to the model file.
    Returns:
        model: The loaded model.
    """
    model = joblib.load(model_path)
    return model

def evaluate_model(model, X, y):
    """
    Evaluate the model using the loaded data.
    Args:
        model: The trained model.
        X (pd.DataFrame): Features for prediction.
        y (pd.Series): True labels.
    Returns:
        dict: A dictionary containing model evaluation metrics.
    """
    predictions = model.predict(X)
    metrics = {
        'Accuracy': accuracy_score(y, predictions),
        'Precision': precision_score(y, predictions, average='macro'),
        'Recall': recall_score(y, predictions, average='macro'),
        'F1 Score': f1_score(y, predictions, average='macro')
    }

    # Displaying and saving the confusion matrix
    conf_matrix = confusion_matrix(y, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('confusion_matrix.png')
    plt.show()

    return metrics

def main():
    data_path = './data/new/Preprocessed_Dataset_5971.csv'
    model_path = './models/spam_classifier_model.pkl'  # Update this to your actual model path

    # Load the data
    X, y = load_preprocessed_data(data_path)

    # Load the model
    model = load_model(model_path)

    # Evaluate the model
    metrics = evaluate_model(model, X, y)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main()
