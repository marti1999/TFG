import numpy as np
import matplotlib.pyplot as plt
import csv


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

    plt.savefig("../results/figures/" + file_name + ".png")
    plt.show()


def save_results_to_csv(model_name, params, dataset_name, type, average, acc, prec, recall, f1, seconds):
    row = [model_name, params, dataset_name, type, average, str(acc), str(prec), str(recall), str(f1), str(seconds)]

    with open(r'../results/results.csv', 'a', newline='') as fd:
        write = csv.writer(fd, delimiter=';')
        write.writerow(row)
