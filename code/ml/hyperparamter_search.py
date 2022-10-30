import optuna

import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm


class Objective(object):
    def __init__(self, x, y):
        self.X = x
        self.y = y

    def __call__(self, trial):
        x, y = self.X, self.y

        classifier_name = trial.suggest_categorical("classifier", ["NB", "RF"])
        if classifier_name == "NB":
            nb_alpha = trial.suggest_float("nb_alpha", 0.00001, 1000, log=True)
            classifier_obj = sklearn.naive_bayes.MultinomialNB(alpha=nb_alpha)

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
            classifier_obj = sklearn.ensemble.RandomForestClassifier(
                max_depth=rf_max_depth, n_estimators=rf_n_estimators,
                min_samples_leaf=rf_min_samples_leaf, min_samples_split=rf_min_samples_split
            )

        score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=5, scoring="recall")
        return score.mean()
