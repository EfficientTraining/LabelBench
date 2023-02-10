import wandb
import numpy as np
import argparse
from torch.utils.data import Subset

from src.dataset import get_dataset
from model import get_model_class
from trainer import PassiveTrainer, get_fns

parser = argparse.ArgumentParser()
parser.add_argument("wandb_name", type=str, default="")
parser.add_argument("project_name", type=str, default="")
parser.add_argument("seed", type=int, default="")
args = parser.parse_args()
wandb_name = args.wandb_name
project_name = args.project_name

api = wandb.Api()

for run_name in ["%s/Active Learning, %s"]:
    runs = api.runs(run_name % (wandb_name, project_name))
    for run in runs:
        config = run.config
        if args.seed != config["seed"]:
            continue
        num_batch = config["num_batch"]
        batch_size = config["batch_size"]
        data_name = config["data"]
        model_name = config["model"]
        if "car" in project_name:
            n_epoch = 15
        elif "voc" in project_name:
            n_epoch = 15
        elif "coco" in project_name:
            n_epoch = 10
        elif "celeb" in project_name:
            n_epoch = 8
        elif data_name == "caltech":
            n_epoch = 15
        elif data_name == "kuzushiji":
            n_epoch = 10
        elif data_name == "cifar10_imb_2":
            n_epoch = 10
        elif data_name == "cifar100_imb_2":
            n_epoch = 10
        elif data_name == "svhn_imb_2":
            n_epoch = 10
        elif data_name == "cifar100_imb_10":
            n_epoch = 10
        else:
            raise Exception("Unknown Dataset")

        print(run.name)
        labeled = -np.ones(num_batch * batch_size, dtype=int)
        for row in run.scan_history(keys=["i", "Label index"]):
            labeled[row["i"]] = row["Label index"]
        assert np.min(labeled) > -1e-8

        train_dataset, val_dataset, test_dataset, multi_label_flag, n_class = get_dataset(data_name, batch_size)
        wandb.init(project="Active Learning, %s" % project_name, entity=wandb_name, name=run.name, config=config)
        model_class = get_model_class(model_name)
        loss_fn, pred_fn, metric = get_fns(multi_label_flag)
        trainer = PassiveTrainer(model_class, n_class, n_epoch, loss_fn, metric, pred_fn)

        for idx in range(1, num_batch + 1):
            num_labeled = idx * batch_size
            model = trainer.train(Subset(train_dataset, labeled[:num_labeled]), None)
            train_preds, train_labels, train_losses, embs = trainer.test(train_dataset, model, ret_emb=True)
            val_preds, val_labels, val_losses = trainer.test(val_dataset, model)
            test_preds, test_labels, test_losses = trainer.test(test_dataset, model)
            wandb.log(
                metric.compute(idx, train_preds, train_labels, train_losses, val_preds, val_labels, val_losses,
                               test_preds, test_labels, test_losses, num_labeled=num_labeled,
                               labeled=labeled[:num_labeled]))

        wandb.finish()
