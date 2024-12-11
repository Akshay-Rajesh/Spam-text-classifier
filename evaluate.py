from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import cross_validate
import mlflow

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and logs metrics to MLflow.
    """
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log metrics in MLflow
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)

    # Print results
    print("Test Set Evaluation:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

def evaluate_cross_validation(model, X_train, y_train, cv=5):
    """
    Performs cross-validation on the training set and logs metrics to MLflow.
    """
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted')
    }

    # Perform cross-validation
    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)

    # Summarize results
    metrics_summary = {
        metric: {
            "mean": cv_results[f"test_{metric}"].mean(),
            "std": cv_results[f"test_{metric}"].std()
        } for metric in scoring.keys()
    }

    # Log cross-validation metrics to MLflow
    for metric, stats in metrics_summary.items():
        mlflow.log_metric(f"cv_{metric}_mean", stats["mean"])
        mlflow.log_metric(f"cv_{metric}_std", stats["std"])

    # Print cross-validation results
    print("Cross-Validation Results:")
    for metric, stats in metrics_summary.items():
        print(f"{metric}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}")

    return metrics_summary
