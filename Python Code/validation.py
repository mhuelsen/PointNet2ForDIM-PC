import numpy
import torch
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np

"""
This python code is part of a master thesis with the title: 
Analysis of deep learning methods for semantic segmentation of photogrammetric point clouds from aerial images

© Markus Hülsen, Matr.-Nr. 6026370
Date: 22.08.2023

This contains functions to plot a ROC curve or a confusion matrix
"""


def ROC_Plot(mean_fpr: torch.Tensor, mean_tpr: torch.Tensor,
             num_classes: int, class_names: list, ex_path: str = None, show: bool = False):
    """
    Function to plot the ROC-curves and save it to a path
    :param mean_fpr: Tensor with False Positiv Rate
    :param mean_tpr: Tensor with True Positiv Rate
    :param num_classes: number of classes
    :param class_names: list witch contains the class names
    :param ex_path: export path. If None it doesn't export the plot.
    :param show: if True the function will show the plot
    """

    for class_idx in range(num_classes):
        fpr = mean_fpr[class_idx, :]
        tpr = mean_tpr[class_idx, :]

        plt.plot(fpr.to(torch.device('cpu')),
                 tpr.to(torch.device('cpu')),
                 lw=2, label=class_names[class_idx])

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, color='lightgrey', ls='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Multiclass')
    plt.legend(loc="lower right")
    if ex_path:
        plt.savefig(ex_path)
    if show:
        plt.show()


def plot_confusion_matrix(conf_mat: numpy.ndarray, class_names: list, ex_path: str = None, show=False):
    """
    Function to plot a confusion matrix
    :param conf_mat: array with the values of the confusion matrix
    :param class_names: list with all class names in the right order
    :param ex_path: Export path where the plot will be stored. If None, it doesn't export the plot
    :param show: if True, it will show the plot
    """
    # normalize confusion matrix
    conf_mat = conf_mat / np.sum(conf_mat, axis=1)[:, np.newaxis]
    # define colormap
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.get_cmap('viridis'))

    # plot adjustments
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()

    # set axis labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # display values inside the cells
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, '{:.2f}%'.format(conf_mat[i, j] * 100),
                     ha="center", color="white" if conf_mat[i, j] < 0.5 else "black")

    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    if ex_path:
        plt.savefig(ex_path)
    if show:
        plt.show()
