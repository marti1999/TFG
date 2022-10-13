from sklearn.tree import DecisionTreeClassifier

def execute_dtc(X_train, y_train, X_test):
    dtc = DecisionTreeClassifier(criterion='entropy', random_state=1)
    dtc.fit(X_train, y_train)
    return dtc.predict(X_test)