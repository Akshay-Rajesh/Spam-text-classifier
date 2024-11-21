from sklearn.naive_bayes import MultinomialNB

def train_model(X_train, y_train):
    # Initialize and train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model


