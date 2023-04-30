import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from timm.loss import LabelSmoothingCrossEntropy
from timm.data.mixup import Mixup

import ALBench.trainer.trainer_impl
from ALBench.skeleton.trainer_skeleton import trainers


def get_trainer(name, trainer_config, dataset, model_fn, model_config, metric, update_embed_dataset_fn):
    return trainers[name](trainer_config, dataset, model_fn, model_config, metric, update_embed_dataset_fn)


def get_fns(trainer_config):
    assert "loss_fn" in trainer_config, "loss_fn not in trainer config."
    assert "pred_fn" in trainer_config, "pred_fn not in trainer config."
    # Loss function
    if trainer_config["loss_fn"] == "Binary Cross Entropy Multi Class":
        trainer_config["loss_fn"] = lambda x, y, weight=None: \
            F.binary_cross_entropy_with_logits(x[:, 1:2], y[:, 1:2],
                                               pos_weight=(None if weight is None else (weight[1] / weight[0])))
    elif trainer_config["loss_fn"] == "Cross Entropy":
        trainer_config["loss_fn"] = F.cross_entropy
    elif trainer_config["loss_fn"] == "LabelSmoothingCrossEntropy":
        def loss_fn(x, target):
            loss = LabelSmoothingCrossEntropy(smoothing=trainer_config["smoothing"])
            return loss(x, torch.argmax(target, dim=-1))

        trainer_config["loss_fn"] = loss_fn

    # Prediction function
    if trainer_config["pred_fn"] == "Sigmoid":
        trainer_config["pred_fn"] = torch.sigmoid
    elif trainer_config["pred_fn"] == "Sigmoid Multi Class":
        trainer_config["pred_fn"] = lambda x: torch.sigmoid(x[:, 1:2])
    elif trainer_config["pred_fn"] == "Softmax":
        trainer_config["pred_fn"] = lambda x: torch.softmax(x, dim=-1)

    # Mixup function
    mixup_active = "mixup" in trainer_config and trainer_config["mixup"] > 0. or \
                     "cutmix" in trainer_config and trainer_config["cutmix"] > 0. 
    # Note we put label smoothing to 0.0 here because it is already done in the loss function.
    def mixup_fn(num_classes):
        if mixup_active:
            return Mixup(mixup_alpha=trainer_config["mixup"] if "mixup" in trainer_config else 0., 
                            cutmix_alpha=trainer_config["cutmix"] if "cutmix" in trainer_config else 0., 
                            label_smoothing=0.0, num_classes=num_classes)
        else:
            return None
    trainer_config["mixup_fn"] = mixup_fn

    return trainer_config


def get_optimizer_fn(trainer_config):
    train_method = trainer_config["trainer_name"].split("_")[0]

    # If we are using sklearn, we don't need to further configure the optimizer.
    if train_method != "sklearn":
        wd = trainer_config["wd"] if "wd" in trainer_config else 0
        if trainer_config["optim_name"] == "Adam":
            if "betas" not in trainer_config:
                def optim_fn(params):
                    return Adam(
                        params, lr=trainer_config["lr"], weight_decay=wd)
            else:
                def optim_fn(params):
                    return Adam(params, lr=trainer_config["lr"], betas=tuple(trainer_config["betas"]),
                                weight_decay=wd)
        elif trainer_config["optim_name"] == "AdamW":
            if "betas" not in trainer_config:
                def optim_fn(params):
                    return AdamW(
                        params, lr=trainer_config["lr"], weight_decay=wd)
            else:
                def optim_fn(params):
                    return AdamW(params, lr=trainer_config["lr"], betas=tuple(trainer_config["betas"]),
                                 weight_decay=wd)
        elif trainer_config["optim_name"] == "SGD":
            nesterov = trainer_config["nesterov"] if "nesterov" in trainer_config else False
            momentum = trainer_config["momentum"] if "momentum" in trainer_config else 0

            def optim_fn(params):
                return SGD(params, lr=trainer_config["lr"], weight_decay=wd, nesterov=nesterov,
                           momentum=momentum)
        else:
            raise ValueError("%s optimizer is unknown" %
                             trainer_config["optim_name"])
        trainer_config["optim_fn"] = optim_fn
    return trainer_config


def get_scheduler_fn(trainer_config):
    if "scheduler_name" in trainer_config:
        if trainer_config["scheduler_name"] == "CosineLR":
            def scheduler_fn(optimizer, total_steps):
                return CosineAnnealingWarmRestarts(optimizer, T_0=trainer_config["warmup_steps"], 
                                                   T_mult=trainer_config["T_multi"] if "T_multi" in trainer_config else 2,
                                                   eta_min=trainer_config["eta_min"] if "eta_min" in trainer_config else 0)
        # Use the customized cosine lr scheduler in wise-FT. (A special case of CosineLR)
        elif trainer_config["scheduler_name"] == "customized_CosineLR":
            def scheduler_fn(optimizer, total_steps):
                return cosine_lr(optimizer, base_lrs=trainer_config["lr"], warmup_length=trainer_config["warmup_steps"],
                                 steps=total_steps)
        elif trainer_config["scheduler_name"] == "StepLR":
            def scheduler_fn(optimizer, total_steps):
                return step_lr(optimizer, base_lr=trainer_config["lr"], step_size=trainer_config["step_size"],
                               gamma=trainer_config["gamma"])
        else:
            raise ValueError("%s scheduler is unknown" %
                             trainer_config["scheduler_name"])
        trainer_config["scheduler_fn"] = scheduler_fn

    return trainer_config





# Modified cosine_lr functions, copy from https://github.com/mlfoundations/wise-ft/blob/master/src/models/utils.py.
def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def step_lr(optimizer, base_lr, step_size, gamma):
    def _lr_adjuster(step):
        for param_group in optimizer.param_groups:
            if (step + 1) % step_size == 0:
                assign_learning_rate(param_group, base_lr * (gamma ** ((step + 1) // step_size)))

    return _lr_adjuster
