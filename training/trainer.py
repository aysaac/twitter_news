import os
import logging

import torch
from models.model_factory import get_model
from utils.factories import (
    get_device,
    get_optimizer,
    get_loss,
)
from training.metrics import Metrics
from training.evaluator import Evaluator
from data_ingestion.dataloader.dataloader_factory import (
    get_dataloaders,
)
import wandb

logger = logging.getLogger("spoof")


class NLP_Trainer:
    """Train a model using the given configuration.
    :param: cfg: Configuration to use.
    """

    def __init__(self, cfg: dict) -> None:

        logger.info("Starting training")

        self._experiment_name = cfg["experiment_name"]
        self._save_last = cfg["model"]["save_last_model"]
        self._save_best = cfg["model"]["save_best_model"]
        self._write_dir = cfg["model"]["write_model_dir"]
        self._epochs = cfg["train_config"]["parameters"]["epochs"]
        self._device = get_device(
            device_type=cfg["device"],
            device_index=cfg.get("device_index", None),
        )
        # self._device=torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        # print(self._device)

        logger.info(f"Meta parameters:")
        logger.info(f"Experiment name: {self._experiment_name}")
        logger.info(f"Save last model: {self._save_last}")
        logger.info(f"Save best model: {self._save_best}")
        logger.info(f"Write directory: {self._write_dir}")
        logger.info(f"Device: {self._device}")
        logger.info(f"Model path: {cfg['model']['resume_model_path']}")
        logger.info(f"Training mode: {cfg['mode']}")
        logger.info(
            "Number of workers: "
            f"{cfg['train_config']['parameters']['num_workers']}"
        )
        logger.info(f"Model kind: {cfg['model']['kind']}")
        logger.info(f"Model path: {cfg['model']['resume_model_path']}")
        logger.info(f"Model args: {cfg['model']['model_args']}")

        logger.info(f"Training parameters:")
        logger.info(f"Epochs: {self._epochs}")
        logger.info(
            f"Batch size: {cfg['train_config']['parameters']['batch_size']}"
        )
        logger.info(
            f"Optimizer params: {cfg['train_config']['parameters']['optimizer']}"
        )

        # Loss function
        logger.info("Creating loss function...")

        criterion = get_loss(loss=cfg["loss"])

        # Model
        logger.info("Creating model...")
        model_args = cfg["model"]["model_args"]
        self._model = get_model(
            kind=cfg["model"]["kind"],
            model_path=cfg["model"]["resume_model_path"],
            **model_args,
        )
        self._mode = cfg["mode"]
        self._model = self._model.to(self._device)

        # Optimizer
        logger.info("Creating optimizer...")
        # if self._mode == 'simultaneous':
        #     self._optimizer1 = get_optimizer(cfg, self._model.clsf_head)
        #     self._optimizer2 = get_optimizer(cfg, self._model.CDCM)
        # else:
        self._optimizer = get_optimizer(cfg, self._model)

        # Dataloaders
        logger.info("Creating dataloaders...")

        dataloaders = get_dataloaders(cfg)
        self._train_loader = dataloaders["train"]
        self._val_loader = dataloaders["validation"]
        self._test_loader = dataloaders["test"]
        self._trainer = Trainer(
            mode=cfg["mode"],
            model=self._model,
            device=self._device,
            criterion=criterion,
            optimizer=self._optimizer,
            dataloader=self._train_loader,
        )
        self._evaluator = Evaluator(
            mode=cfg["mode"],
            model=self._model,
            device=self._device,
            criterion=criterion,
            dataloader=self._val_loader,
            wandb_logger=False,
            type="Validation",
        )
        self._tester = Evaluator(
            mode=cfg["mode"],
            model=self._model,
            device=self._device,
            criterion=criterion,
            dataloader=self._test_loader,
            wandb_logger=True,
            type="Test",
        )
        self._full_test = Evaluator(
            mode=cfg["mode"],
            model=self._model,
            device=self._device,
            criterion=criterion,
            dataloader=dataloaders["full"],
            wandb_logger=True,
            type="Test",
        )

    def _get_write_path(self, kind: str) -> str:
        """Returns a path to save a model.
        :param kind: One of {best, last}.
        """
        if kind == "best":
            write_name = self._experiment_name + "_best.pt"
            return os.path.join(self._write_dir, write_name)
        elif kind == "last":
            write_name = self._experiment_name + "_last.pt"
            return os.path.join(self._write_dir, write_name)
        else:
            raise ValueError()

    def train(self):
        best_acc = 0
        best_loss = 50000
        for epoch in range(self._epochs):
            print(f"Epoch {epoch + 1}/{self._epochs}")
            print("-" * 10)
            logger.info("Epoch: " + str(epoch))
            self._trainer.train()
            self._evaluator.evaluate()
            if self._mode == "depth_map":
                if self._evaluator.metrics < best_loss:
                    write_path = self._get_write_path(kind="best")
                    torch.save(self._model.state_dict(), write_path)
                    best_loss = self._evaluator.metrics
            else:
                if self._evaluator.metrics.accuracy > best_acc:
                    write_path = self._get_write_path(kind="best")
                    torch.save(self._model.state_dict(), write_path)
                    best_acc = self._evaluator.metrics.accuracy
        if self._save_last:
            write_path = self._get_write_path(kind="last")
            torch.save(self._model.state_dict(), write_path)

        self._tester.evaluate()

    def test(self):
        self._full_test.evaluate()

    def partial_test(self):
        self._tester.evaluate()


class Trainer:
    """Inteface for training.
    param mode: One of {standard, fusion}
    """

    def __init__(
            self,
            mode,
            model,
            device,
            criterion,
            optimizer,
            dataloader,
    ):
        self._epoch = 0
        self._batch = 0
        self._mode = mode
        self._model = model
        self._device = device
        self._criterion = criterion
        self._optimizer = optimizer
        self._dataloader = dataloader
        # self._max_itea = 0

    def train(self):
        logger.info("Starting training step")
        if self._mode == "standard":
            self._standard_train()
        elif self._mode == 'BERT':
            self._BERT_train()
        else:
            raise NotImplementedError()

    def _standard_train(self):
        self._model.train()
        correct_predictions = 0
        cum_loss = 0
        for i, sample in enumerate(self._dataloader):
            logger.info(f"Processing batch {i}")
            if i % 50 == 0:
                print(f"Processing batch {i}")
            input = sample["embeddings"]
            target = sample["target"]
            input = input.to(self._device)
            target = target.to(self._device)
            n_examples = len(target)
            self._optimizer.zero_grad()
            output = self._model(input)
            _, preds = torch.max(output, dim=1)  # se toma el softmax
            correct_predictions = torch.sum(preds == target)
            wandb.log({'Train accuracy': correct_predictions.double() / n_examples})
            # breakpoint()
            loss = self._criterion(output, target)

            wandb.log({'Batch train loss ': loss})

            loss.backward()
            self._optimizer.step()

            self._batch += 1

            if i == 0:
                cum_loss = loss
            else:
                cum_loss += loss
            # metrics.update(output, target)
        wandb.log({"Cumulative train loss": cum_loss})
        self._epoch += 1

    def _BERT_train(self):
        self._model.train()

        cum_loss = 0
        for i, sample in enumerate(self._dataloader):
            logger.info(f"Processing batch {i}")
            if i % 50 == 0:
                print(f"Processing batch {i}")
            input_ids = sample["input_ids"].to(self._device)
            attention_mask = sample["attention_mask"].to(self._device)
            targets = sample["targets"].to(self._device)  # labes
            # text=sample["texts"]
            n_examples = len(targets)
            self._optimizer.zero_grad()
            outputs = self._model(input_ids=input_ids,
                                  attention_mask=attention_mask)  # propaga el momento haciadelante /predictions
            # breakpoint()
            _, preds = torch.max(outputs, dim=1)  # se toma el softmax
            correct_predictions = torch.sum(preds == targets)
            loss = self._criterion(outputs, targets)  # se mide la perdida
            wandb.log({'Batch train loss ': loss})
            wandb.log({'Train accuracy': correct_predictions.double() / n_examples})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._optimizer.step()

            self._batch += 1

            if i == 0:
                cum_loss = loss
            else:
                cum_loss += loss
            # metrics.update(output, target)
        wandb.log({"Cumulative train loss": cum_loss})

        self._epoch += 1
