from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer,confusion_matrix
from sklearn.model_selection import cross_validate
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set, logs metrics and confusion matrix to MLflow.
    """
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

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

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    # Save the figure and log as artifact
    cm_filename = "confusion_matrix.png"
    plt.savefig(cm_filename)
    plt.close()  # Close the plot to avoid displaying it during execution
    mlflow.log_artifact(cm_filename)

def evaluate_cross_validation(model, X_train, y_train, cv=5):
    """
    Performs cross-validation on the training set and logs metrics for each fold to MLflow.
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

    # Log metrics for each fold to MLflow
    for fold_idx in range(cv):
        mlflow.log_metric(f"fold_{fold_idx+1}_accuracy", cv_results['test_accuracy'][fold_idx])
        mlflow.log_metric(f"fold_{fold_idx+1}_precision", cv_results['test_precision'][fold_idx])
        mlflow.log_metric(f"fold_{fold_idx+1}_recall", cv_results['test_recall'][fold_idx])
        mlflow.log_metric(f"fold_{fold_idx+1}_f1", cv_results['test_f1'][fold_idx])

    # Print cross-validation results for each fold
    print("Cross-Validation Results (Per Fold):")
    for fold_idx in range(cv):
        print(f"Fold {fold_idx+1}:")
        print(f"  Accuracy: {cv_results['test_accuracy'][fold_idx]:.4f}")
        print(f"  Precision: {cv_results['test_precision'][fold_idx]:.4f}")
        print(f"  Recall: {cv_results['test_recall'][fold_idx]:.4f}")
        print(f"  F1 Score: {cv_results['test_f1'][fold_idx]:.4f}")

    return cv_results

