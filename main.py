from evaluate import evaluate_model, evaluate_cross_validation
from data_loader import load_data
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
import mlflow

def tune_hyperparameters(X_train, y_train):
    """
    Performs hyperparameter tuning using GridSearchCV.
    """
    # Define the parameter grid for MultinomialNB
    param_grid = {
        "alpha": [0.1, 0.5, 1.0, 1.5, 2.0],  # Smoothing parameter
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=MultinomialNB(),
        param_grid=param_grid,
        scoring="f1_weighted",  # Optimize for weighted F1-score
        cv=5,  # 5-fold cross-validation
        verbose=1,
        n_jobs=-1  # Use all available cores
    )

    # Perform the grid search
    grid_search.fit(X_train, y_train)

        # Log the best hyperparameters in MLflow
    best_params = grid_search.best_params_
    for param_name, param_value in best_params.items():
        mlflow.log_param(param_name, param_value)

    # Log the best cross-validation score
    mlflow.log_metric("best_cv_f1_score", grid_search.best_score_)

    return grid_search

def main():
    # Start the MLFlow experiment
    mlflow.set_experiment("SMS_Spam_Classificator")

    with mlflow.start_run(run_name="Naive Bayes with Hyperparameter Tuning(logged in mlflow)"):
        # Load the data and vectorizer
        X_train, X_test, y_train, y_test, vectorizer = load_data()

        # Log vectorizer details
        mlflow.log_param("vectorizer", str(vectorizer))

        # Perform hyperparameter tuning
        print("Tuning hyperparameters...")
        grid_search = tune_hyperparameters(X_train, y_train)

        # Log the best parameters and score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_f1_score", best_score)

        print(f"Best Parameters: {best_params}")
        print(f"Best Cross-Validation F1 Score: {best_score:.4f}")

        # Train the model with the best parameters
        model = grid_search.best_estimator_

        # Perform cross-validation on the best model
        print("Performing cross-validation with the best model...")
        evaluate_cross_validation(model, X_train, y_train, cv=5)

        # Evaluate the model on the test set
        print("Evaluating the best model on test data...")
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
