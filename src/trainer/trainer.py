import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

import src.trainer.trainer_impl
from src.skeleton.trainer_skeleton import trainers


def get_trainer(name, trainer_config, dataset, model_fn, model_config, metric):
    return trainers[name](trainer_config, dataset, model_fn, model_config, metric)


def get_fns(trainer_config):
    assert "loss_fn" in trainer_config, "loss_fn not in trainer config."
    assert "pred_fn" in trainer_config, "pred_fn not in trainer config."
    # Loss function
    if trainer_config["loss_fn"] == "Binary Cross Entropy":
        trainer_config["loss_fn"] = F.binary_cross_entropy_with_logits
    elif trainer_config["loss_fn"] == "Cross Entropy":
        trainer_config["loss_fn"] = F.cross_entropy

    # Prediction function
    if trainer_config["pred_fn"] == "Sigmoid":
        trainer_config["pred_fn"] = torch.sigmoid
    elif trainer_config["pred_fn"] == "Softmax":
        trainer_config["pred_fn"] = lambda x: torch.softmax(x, dim=-1)

    return trainer_config


def get_optimizer_fn(trainer_config):
    wd = trainer_config["wd"] if "wd" in trainer_config else 0
    if trainer_config["optim_name"] == "Adam":
        if "betas" not in trainer_config:
            optim_fn = lambda params: Adam(params, lr=trainer_config["lr"], weight_decay=wd)
        else:
            optim_fn = lambda params: Adam(params, lr=trainer_config["lr"], betas=tuple(trainer_config["betas"]),
                                        weight_decay=wd)
    elif trainer_config["optim_name"] == "SGD":
        nesterov = trainer_config["nesterov"] if "nesterov" in trainer_config else False
        momentum = trainer_config["momentum"] if "momentum" in trainer_config else 0
        optim_fn = lambda params: SGD(params, lr=trainer_config["lr"], weight_decay=wd, nesterov=nesterov,
                                   momentum=momentum)
    else:
        raise ValueError("%s optimizer is unknown" % trainer_config["optim_name"])
    trainer_config["optim_fn"] = optim_fn
    return trainer_config
