import copy
from datetime import datetime
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import csv
import json

import optuna.visualization


class Results:
    def __init__(self, model_name=None, params=None, dataset_name=None, type=None, average=None, acc=None, prec=None,
                 recall=None, f1=None, seconds=None, hyperparameter_search=False, datetime=None):
        self.model_name = model_name
        self.params = params
        self.dataset_name = dataset_name
        self.type = type
        self.average = average
        self.acc = acc
        self.prec = prec
        self.recall = recall
        self.f1 = f1
        self.seconds = seconds
        self.hyperparameter_search = hyperparameter_search
        self.datetime = datetime

    def get_csv_row(self):
        return [self.model_name, self.params, self.dataset_name, self.type, self.average, str(self.acc), str(self.prec),
                str(self.recall), str(self.f1), str(self.seconds), str(self.hyperparameter_search), str(self.datetime)]


def bar_plot_multiple_column(models, type1, type2, type1desc, type2desc, file_name):
    x_axis = np.arange(len(models))

    plt.figure(figsize=(9, 7))

    # subplot
    plt.subplot(2, 2, 1)
    plt.bar(x_axis - 0.2, type1.acc, width=0.4, label=type1desc)
    plt.bar(x_axis + 0.2, type2.acc, width=0.4, label=type2desc)
    plt.xticks(x_axis, models)
    plt.legend()
    plt.title('Accuracy')
    plt.ylim([0, 1.2])

    # subplot
    plt.subplot(2, 2, 2)
    plt.bar(x_axis - 0.2, type1.prec, width=0.4, label=type1desc)
    plt.bar(x_axis + 0.2, type2.prec, width=0.4, label=type2desc)
    plt.xticks(x_axis, models)
    plt.legend()
    plt.title('Precision')
    plt.ylim([0, 1.2])

    # subplot
    plt.subplot(2, 2, 3)
    plt.bar(x_axis - 0.2, type1.recall, width=0.4, label=type1desc)
    plt.bar(x_axis + 0.2, type2.recall, width=0.4, label=type2desc)
    plt.xticks(x_axis, models)
    plt.legend()
    plt.title('Recall')
    plt.ylim([0, 1.2])

    # subplot
    plt.subplot(2, 2, 4)
    plt.bar(x_axis - 0.2, type1.f1, width=0.4, label=type1desc)
    plt.bar(x_axis + 0.2, type2.f1, width=0.4, label=type2desc)
    plt.xticks(x_axis, models)
    plt.legend()
    plt.title('F1 Score')
    plt.ylim([0, 1.2])

    plt.savefig("../results/figures/single_tests/" + file_name + ".png")
    plt.show()

def plot_proba_histogram(proba, dataset=None, type=None, model=None):
    # plt.hist(proba, bins=10, range=[0.5, 1])
    # plt.title(model)
    # plt.ylabel("Count")
    # plt.xlabel("Prediction probability")
    # plt.savefig("../results/figures/proba/"+ dataset + "_"+type+"_"+model+".png")

    sns.histplot(data=proba, bins=10, binrange=[0.5, 1], stat="percent")
    plt.ylim(0, 110)
    plt.title(model + " "+type)
    plt.ylabel("Count")
    plt.xlabel("Prediction probability")
    plt.savefig("../results/figures/proba/" + dataset + "_" + type + "_" + model + ".png")
    plt.clf()

def plot_optuna_metric_histogram(study, dataset, type):
    metric_list = []
    for t in study.trials:
        metric_list.append(t.value)
    plt.hist(metric_list, bins=20, range=[0, 1])
    plt.title("Hyperparameter search results: " + study.best_params['classifier'])
    plt.ylabel("Count")
    plt.xlabel("Recall (macro)")
    plt.savefig(
        "../results/figures/optuna_tests/" + dataset + "_" + type + "_histogram_" + study.best_params[
            'classifier'] + ".png")
    plt.show()


def parse_optuna_trials(trials, file_name, type, scoring):
    results_list = []
    for trial in trials:
        r = Results()
        r.model_name = trial.params['classifier']
        params = copy.copy(trial.params)
        params.pop('classifier')
        r.params = json.dumps(params)
        r.dataset_name = file_name
        r.type = type
        r.average = scoring
        r.f1 = trial.value
        r.seconds = trial.duration.total_seconds()
        r.hyperparameter_search = True
        r.datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results_list.append(r)
    return results_list


# def save_results_to_csv(model_name, params, dataset_name, type, average, acc, prec, recall, f1, seconds):
#     row = [model_name, params, dataset_name, type, average, str(acc), str(prec), str(recall), str(f1), str(seconds)]
#
#     with open(r'../results/results.csv', 'a', newline='') as fd:
#         write = csv.writer(fd, delimiter=';')
#         write.writerow(row)
#
def save_results_to_csv(results):
    row = results.get_csv_row()

    with open(r'../results/results.csv', 'a', newline='') as fd:
        write = csv.writer(fd, delimiter=';')
        write.writerow(row)
