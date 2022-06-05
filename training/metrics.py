import torch
import wandb
import numpy as np

def np_sigmoid(x):
    return 1/(1 + np.exp(-x))
class Metrics:
    def __init__(self):
        self._TP = 0
        self._TN = 0
        self._FP = 0
        self._FN = 0
        self.matrix=None


    @property
    def accuracy(self):
        correct = self._TP + self._TN
        total = self._TP + self._TN + self._FP + self._FN
        return correct / total

    @property
    def APCER(self):
        """APCER: Attack Presentation Classification Error Rate"""
        try:
            return self._FN / (self._TP + self._FN)
        except ZeroDivisionError:
            return 1.0

    @property
    def BPCER(self):
        """Bona Fide Presentation Classification Error Rate"""
        try:
            return self._FP / (self._TN + self._FP)
        except ZeroDivisionError:
            return 1.0

    def update(
        self, output: np.array, target: np.array, is_prob: bool = True
    ) -> None:
        if is_prob:
            output = np.round(np_sigmoid(output))
        self._update_TP(output, target)
        self._update_TN(output, target)
        self._update_FP(output, target)
        self._update_FN(output, target)
        self._update_CM(output, target)
    def _update_TP(self, output, target):
        self._TP += ((output == 1.0) & (target == 1.0)).sum().item()

    def _update_TN(self, output, target):
        self._TN += ((output == 0.0) & (target == 0.0)).sum().item()

    def _update_FP(self, output, target):
        self._FP += ((output == 1.0) & (target == 0.0)).sum().item()

    def _update_FN(self, output, target):
        self._FN += ((output == 0.0) & (target == 1.0)).sum().item()
    def _update_CM(self, output, target):
        names=['Live','Spoof']
        conf_matrix=wandb.plot.confusion_matrix(probs=None,
                                    y_true=target,
                                    preds=output,
                                    class_names=names
                                    )
        self.matrix=conf_matrix
