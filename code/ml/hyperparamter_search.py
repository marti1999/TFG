import optuna

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

import sklearn.model_selection


class Objective(object):
    def __init__(self, x, y, average='macro'):
        self.X = x
        self.y = y
        self.score = average

    def __call__(self, trial):
        x, y = self.X, self.y

        # classifier_name = trial.suggest_categorical("classifier", ["NB", "RF"])
        classifier_name = trial.suggest_categorical("classifier", ["NB", "RF", "DTC", "KNN"])
        if classifier_name == "NB":
            nb_alpha = trial.suggest_float("nb_alpha", 0.00001, 1000, log=True)
            classifier_obj = MultinomialNB(alpha=nb_alpha)

        if classifier_name == "RF":
            # param = {
            #     "max_depth": trial.suggest_int("rf_max_depth", 5, 100, log=True),
            #     "n_estimators": trial.suggest_int("rf_n_estimators", 5, 30, log=True),
            #     "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 5, log=False),
            #     "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10, log=False)
            # }

            rf_max_depth = trial.suggest_int("rf_max_depth", 5, 100, log=True)
            rf_n_estimators = trial.suggest_int("rf_n_estimators", 5, 30, log=True)
            rf_min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 5, log=False)
            rf_min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 10, log=False)
            classifier_obj = RandomForestClassifier(
                max_depth=rf_max_depth, n_estimators=rf_n_estimators,
                min_samples_leaf=rf_min_samples_leaf, min_samples_split=rf_min_samples_split
            )

        if classifier_name == "DTC":
            dtc_max_depth = trial.suggest_int("dtc_max_depth", 5, 100, log=True)
            dtc_min_samples_leaf = trial.suggest_int("dtc_min_samples_leaf", 1, 50, log=False)
            dtc_criterion = trial.suggest_categorical("dtc_criterion", ["gini", "entropy"])
            dtc_min_samples_split = trial.suggest_int("dtc_min_samples_split", 2, 30, log=False)
            classifier_obj = DecisionTreeClassifier(
                max_depth=dtc_max_depth, min_samples_leaf=dtc_min_samples_leaf,
                min_samples_split=dtc_min_samples_split, criterion=dtc_criterion
            )

        if classifier_name == "SVM":
            svc_kernel = trial.suggest_categorical("svc_kernel", ["linear", "poly", "rbf"])
            svc_gamma = trial.suggest_float("svc_gamma", 0.001, 100, log=True)
            svc_c = trial.suggest_float("svc_c", 0.01, 1000, log=True)
            svc_degree = trial.suggest_int("svc_degree", 1, 5, log=False)
            classifier_obj = svm.SVC(
                kernel=svc_kernel, gamma=svc_gamma, C=svc_c, degree=svc_degree
            )

        if classifier_name == "KNN":
            knn_n_neighbours = trial.suggest_int("knn_n_neighbours", 3, 9, log=False)
            knn_p = trial.suggest_int("knn_p", 1, 2, log=False)
            knn_weight = trial.suggest_categorical("knn_weight", ["uniform", "distance"])
            classifier_obj =  KNeighborsClassifier(
                n_neighbors=knn_n_neighbours, p=knn_p, weights=knn_weight
            )





        score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=5,
                                                        scoring="recall_" + self.score)
        return score.mean()
