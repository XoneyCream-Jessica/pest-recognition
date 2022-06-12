import matplotlib.pyplot as plt
import common.file as common_file
import os
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import itertools

def plot_learning_curve(loss_record, title='', save=True):
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['val'])]
    plt.figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['val'], c='tab:cyan', label='val')
    plt.ylim(0.0, 3.)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    if save:
        plt.savefig(os.path.join(common_file.get_log_path(), "learning_" + title + ".png"))
    plt.show()


def plot_acc_curve(acc_record, title='', save=True):
    total_steps = len(acc_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(acc_record['train']) // len(acc_record['val'])]
    plt.figure(figsize=(6, 4))
    plt.plot(x_1, acc_record['train'], c='tab:red', label='train')
    plt.plot(x_2, acc_record['val'], c='tab:cyan', label='val')
    plt.ylim(0.0, 1.)
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.title('Acc curve of {}'.format(title))
    plt.legend()
    if save:
        plt.savefig(os.path.join(common_file.get_log_path(), "acc_" + title + ".png"))
    plt.show()


def plot_confusion_matrix(cm, labels, title='', save=True):
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f' if cm[i, j] > 0 else '.0f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save:
        plt.savefig(os.path.join(common_file.get_log_path(), "cm_" + title + ".png"))
    plt.show()
