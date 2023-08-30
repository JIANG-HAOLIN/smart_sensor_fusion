from sklearn.metrics import confusion_matrix
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os
from typing import Optional

def plot_confusion_matrix(outs: np.ndarray, labels: np.ndarray, save_pth: Optional[str] = None):
    """plot and save the confusion matrix
    Args:
        outs - the output of inference
        labels - the ground truth
        save_pth - where to save the confusion matrix pic
    """
    accu = (outs == labels).mean()
    conf_mtr = confusion_matrix(outs, labels)
    plt.figure(figsize=(10, 10))
    sb.set(font_scale=1.2)
    sb.heatmap(conf_mtr, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.ylabel('Predicted Labels')
    plt.xlabel('True Labels')
    plt.title(f'Confusion Matrix(Accuracy:{int(accu * 100) / 100})')
    plt.xticks(np.arange(10), labels=range(10), rotation=0)
    plt.yticks(np.arange(10), labels=range(10), rotation=0)
    if save_pth is not None:
        plt.savefig(os.path.join(save_pth,'confusion_matrix.png'), bbox_inches='tight')
    plt.show()