import json
import time
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
    start = time.perf_counter()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    elapsed = time.perf_counter() - start
    return y_pred, get_params_string(nb), elapsed


def execute_dtc(X_train, y_train, X_test):
    dtc = DecisionTreeClassifier(criterion='entropy', random_state=1)
    start = time.perf_counter()
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    elapsed = time.perf_counter() - start
    return y_pred, get_params_string(dtc), elapsed


def execute_rf(X_train, y_train, X_test):
    rf = RandomForestClassifier(n_jobs=-1)
    start = time.perf_counter()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    elapsed = time.perf_counter() - start
    return y_pred, get_params_string(rf), elapsed


def execute_svm(X_train, y_train, X_test):
    cls = svm.SVC(kernel='linear')
    start = time.perf_counter()
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    elapsed = time.perf_counter() - start
    return y_pred, get_params_string(cls), elapsed


def execute_knn(X_train, y_train, X_test):
    knn = KNeighborsClassifier()
    start = time.perf_counter()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    elapsed = time.perf_counter() - start
    return y_pred, get_params_string(knn), elapsed
