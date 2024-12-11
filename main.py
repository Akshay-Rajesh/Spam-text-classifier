from evaluate import evaluate_model, evaluate_cross_validation
from data_loader import load_data
from models.naive_bayes import train_model, cross_validate_model
import mlflow
from sklearn.naive_bayes import MultinomialNB

def main():
    # Start the MLFlow experiment
    mlflow.set_experiment("SMS_Spam_Classificator")

    with mlflow.start_run(run_name="Run 4 Naive Bayes (with cross-validation)"):
        # Load the data and vectorizer
        X_train, X_test, y_train, y_test, vectorizer = load_data()

        # Log vectorizer details
        mlflow.log_param("vectorizer", str(vectorizer))

        # Define the Naive Bayes model
        model = MultinomialNB()

        # Perform cross-validation
        print("Performing cross-validation...")
        evaluate_cross_validation(model, X_train, y_train, cv=5)

        # Train the model on the entire training set
        model.fit(X_train, y_train)

        # Evaluate the model on the test set
        print("Evaluating model on test data...")
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
