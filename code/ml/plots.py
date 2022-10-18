
import numpy as np
import matplotlib.pyplot as plt


def bar_plot_multiple_column(models, type1, type2, type1desc, type2desc):
    x_axis = np.arange(len(models))


    # subplot
    plt.subplot(2,2,1)
    plt.bar(x_axis - 0.2, type1.acc, width=0.4, label=type1desc)
    plt.bar(x_axis + 0.2, type2.acc, width=0.4, label=type2desc)
    plt.xticks(x_axis, models)
    plt.legend()
    plt.title('Accuracy')
    plt.ylim([0, 1.2])

    #subplot
    plt.subplot(2, 2, 2)
    plt.bar(x_axis - 0.2, type1.prec, width=0.4, label=type1desc)
    plt.bar(x_axis + 0.2, type2.prec, width=0.4, label=type2desc)
    plt.xticks(x_axis, models)
    plt.legend()
    plt.title('Precision')
    plt.ylim([0, 1.2])

    #subplot
    plt.subplot(2, 2, 3)
    plt.bar(x_axis - 0.2, type1.recall, width=0.4, label=type1desc)
    plt.bar(x_axis + 0.2, type2.recall, width=0.4, label=type2desc)
    plt.xticks(x_axis, models)
    plt.legend()
    plt.title('Recall')
    plt.ylim([0, 1.2])

    #subplot
    plt.subplot(2, 2, 4)
    plt.bar(x_axis - 0.2, type1.f1, width=0.4, label=type1desc)
    plt.bar(x_axis + 0.2, type2.f1, width=0.4, label=type2desc)
    plt.xticks(x_axis, models)
    plt.legend()
    plt.title('F1 Score')
    plt.ylim([0, 1.2])

    plt.show()
