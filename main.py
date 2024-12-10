from evaluate import evaluate_model
from data_loader import load_data
from models.naive_bayes import train_model
import mlflow

def main():
    # Start the MLFlow experiment
    mlflow.set_experiment("SMS_Spam_Classificator")

    with mlflow.start_run(run_name="Second run (preprocessd data)"):  # Custom run name here
        # Load the data and vectorizer
        X_train, X_test, y_train, y_test, vectorizer = load_data()

        # Train the model
        model = train_model(X_train, y_train)

        # Log model parameters
        mlflow.log_param("model_type", "MultinomialNB")

        # Log vectorizer details
        mlflow.log_param("vectorizer", str(vectorizer))

        # Evaluate the model and log metrics
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
