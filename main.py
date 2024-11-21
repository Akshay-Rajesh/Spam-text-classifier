from evaluate import evaluate_model
from data_loader import load_data
import mlflow

def main():
    # Start the MLFlow experiment
    mlflow.set_experiment("SMS_Spam_Classificator")

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = load_data()

                # Train the model
        # model = ...... model module missing .....
        
        # Evaluation of model
        evaluate_model(model, X_test, y_test)

        print(X_train)


if __name__ == "__main__":
    main()