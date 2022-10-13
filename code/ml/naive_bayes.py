from sklearn.naive_bayes import MultinomialNB

def execute_nb(X_train, y_train, X_test):
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    return nb.predict(X_test)