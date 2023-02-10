import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

import src.trainer.trainer_impl
from src.skeleton.trainer_skeleton import trainers


def get_trainer(name):
    return trainers[name]


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


def get_optimizer(optim_name, lr, weight_decay=0, betas=None, momentum=0, nesterov=False):
    if optim_name == "Adam":
        if betas is None:
            optim = lambda params: Adam(params, lr=lr, weight_decay=weight_decay)
        else:
            optim = lambda params: Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif optim_name == "SGD":
        optim = lambda params: SGD(params, lr=lr, weight_decay=weight_decay, nesterov=nesterov, momentum=momentum)
    else:
        raise ValueError("%s optimizer is unknown" % optim_name)
    return optim
