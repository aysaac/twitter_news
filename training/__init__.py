import os
import logging
import copy
import yaml

import wandb
from trainer import NLP_Trainer
import shutil


def main():

    confing_path=''
    # Read configuration file
    with open(confing_path, "r") as yaml_stream:
        config = yaml.load(yaml_stream, Loader=yaml.FullLoader)

    optimizer_config = config["train_config"]["parameters"]["optimizer"]
    parameters_config = copy.copy(config["train_config"]["parameters"])
    parameters_config.pop("optimizer")
    dataset_configs = config["train_config"]["datasets"].keys()
    wandb_config = {
        "Model": config["model"]["kind"],
        "mode": config["mode"],
        "Datasets": list(dataset_configs),
    }
    wandb_config.update(optimizer_config)
    wandb_config.update(parameters_config)

    wandb.init(project="NLP feminism", config=wandb_config,resume=False)
    # wandb.init(project="Spoof detection dev", config=wandb_config,resume=True)
    record_name = (
        os.environ["CONFIG_PATH"].split("\\")[-1][:-4]
        + config["experiment_name"]
        + "-"
        + wandb.run.name
        + ".yml"
    )
    config["experiment_name"] = config["experiment_name"] + "-" + wandb.run.name
    wandb.run.name = config["experiment_name"]
    logging_dir = config["logging"]["directory"]
    experiment_name = config["experiment_name"]
    logging_file = experiment_name + ".log"
    record_path = os.path.join(os.environ["RECORD_PATH"], record_name)
    shutil.copy(os.environ["CONFIG_PATH"], record_path)
    spoof_logger = logging.getLogger("spoof")
    spoof_handler = logging.FileHandler(os.path.join(logging_dir, logging_file))
    spoof_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(funcName)s] %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S %z",
    )
    spoof_handler.setFormatter(spoof_formatter)
    spoof_logger.addHandler(spoof_handler)
    spoof_logger.setLevel(logging.INFO)
    spoof_logger.propagate = False
    if config['run']=='train':
        execute_experiment(config)
    elif config['run']=='eval':
        test_experiment(config)
    elif config['run']=='partial_eval':
        partial_test(config)
    else:
        raise NotImplementedError()

    # wandb.save(os.path.join(logging_dir, logging_file))
    wandb.finish()


def execute_experiment(config):
    NLP_Trainer(config).train()
def test_experiment(config):
    NLP_Trainer(config).test()
def partial_test(config):
    NLP_Trainer(config).partial_test()

# %%
if __name__ == "__main__":
    main()
