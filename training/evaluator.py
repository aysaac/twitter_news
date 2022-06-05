import logging

# from statistics import mean
import numpy as np
import pandas as pd
import torch
from skimage.measure import compare_ssim
# import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from .metrics import Metrics
def np_sigmoid(x):
    return 1/(1 + np.exp(-x))

logger = logging.getLogger("spoof")


class Evaluator:
    def __init__(
        self, mode, model, device, criterion, dataloader, type, wandb_logger
    ):
        self._epoch = 0
        self._mode = mode
        self._model = model
        self._device = device
        self._dataloader = dataloader
        self._criterion = criterion
        self._type = type
        self.wandb = wandb_logger

    def evaluate(self):
        logger.info(f"Starting {self._type} step")
        if self._mode == "standard":
            self._standard_eval()
        elif self._mode == "siamese":
            self._siamese_eval()
        elif self._mode == "triplet":
            self._p = torch.load("triplet_positive_centroid.pt").cuda()
            self._n = torch.load("triplet_negative_centroid.pt").cuda()
            self._triplet_eval()
        elif self._mode == "fusion":
            self._fusion_eval()
        elif self._mode == "depth_map":
            self._depth_eval()
        elif self._mode == "simultaneous":
            self._simul_eval()
        else:
            raise ValueError(f"Unsupported model: {self._mode}")
    def _standard_eval(self):
        self._model.eval()
        with torch.no_grad():
            self.metrics = Metrics()

            targets=[]
            preds=[]
            if self.wandb:
                table = wandb.Table(columns=['Input', 'Prediction', 'Target', 'Assessment'])
            for i, sample in enumerate(self._dataloader, 1):
                logger.info(f"Processing batch {i}")
                input = sample["img"].to(self._device)
                target = sample["target"].to(self._device)
                output = self._model(input)
                target=target.cpu().detach().numpy().reshape(len(target))
                output=output.cpu().detach().numpy().reshape(len(output))
                input=input.cpu().detach().numpy()
                input = np.moveaxis(input, 1, 3)
                output=np.round(output)
                targets.append(target)
                preds.append(output)
                if self.wandb:
                    output=np.round(np_sigmoid(output))
                    for x in range(len(target)):
                        table.add_data(wandb.Image(input[x]),output[x],target[x],output[x]==target[x])



            preds=np.concatenate(preds)
            targets=np.concatenate(targets)

            self.metrics.update(preds, targets)

            wandb.log(
                {
                    self._type + " " + "TP": self.metrics._TP,
                    self._type + " " + "TN": self.metrics._TN,
                    self._type + " " + "FP": self.metrics._FP,
                    self._type + " " + "FN": self.metrics._FN,
                    self._type + " " + "Accuracy": self.metrics.accuracy,
                    self._type + " " + "APCER": self.metrics.APCER,
                    self._type + " " + "BPCER": self.metrics.BPCER,

                }
            )
        if self.wandb:
            wandb.log({self._type + " " + "Conf Matrix": self.metrics.matrix})
            wandb.log({self._type + " " + "Pred analisi": table})
        self._epoch += 1
