from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


def execute_nb(X_train, y_train, X_test):
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    return nb.predict(X_test)

def execute_dtc(X_train, y_train, X_test):
    dtc = DecisionTreeClassifier(criterion='entropy', random_state=1)
    dtc.fit(X_train, y_train)
    return dtc.predict(X_test)

def execute_rf(X_train, y_train, X_test):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    return rf.predict(X_test)

def execute_svm(X_train, y_train, X_test):
    cls = svm.SVC(kernel='linear')
    cls.fit(X_train, y_train)
    return cls.predict(X_test)