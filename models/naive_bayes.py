from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

def train_model(X_train, y_train):
    """
    Trains the Naive Bayes model on the given training data.
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def cross_validate_model(X, y, cv=5):
    """
    Performs cross-validation on the Naive Bayes model.
    """
    model = MultinomialNB()
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted')
    }
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
    return cv_results


