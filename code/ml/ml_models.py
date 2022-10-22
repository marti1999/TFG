import json

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

def get_params_string(model):
    params = model.get_params()
    return json.dumps(params)
def execute_nb(X_train, y_train, X_test):
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    return nb.predict(X_test), get_params_string(nb)


def execute_dtc(X_train, y_train, X_test):
    dtc = DecisionTreeClassifier(criterion='entropy', random_state=1)
    dtc.fit(X_train, y_train)
    return dtc.predict(X_test), get_params_string(dtc)


def execute_rf(X_train, y_train, X_test):
    rf = RandomForestClassifier(n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf.predict(X_test), get_params_string(rf)


def execute_svm(X_train, y_train, X_test):
    cls = svm.SVC(kernel='linear')
    cls.fit(X_train, y_train)
    return cls.predict(X_test), get_params_string(cls)


def execute_knn(X_train, y_train, X_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    return knn.predict(X_test), get_params_string(knn)
