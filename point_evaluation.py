import argparse
import json
import os
import pickle
import numpy as np
import torch
import wandb
from LabelBench.trainer.trainer import (get_fns, get_optimizer_fn, get_scheduler_fn, get_trainer)
from LabelBench.dataset.datasets import get_dataset
from LabelBench.metric.metrics import get_metric
from LabelBench.model.model import get_model_fn
from LabelBench.corrupter.corrupter import get_corrupter_fn


def retrieve_run(seed, wandb_name, project_name, dataset_name, embed_model_config, classifier_model_config,
                 strategy_config, trainer_config, corrupter_config):
    path_name = "./results/%s/bs_%d/embed_%s/model_%s/trainer_%s/%s_%d.pkl" % (
        dataset_name, int(project_name.split("=")[-1]), embed_model_config, classifier_model_config, trainer_config,
        strategy_config, seed)
    if os.path.exists(path_name):
        with open(path_name, "rb") as file:
            _, idxs = pickle.load(file)
        return np.array(idxs, dtype=int)
    else:
        api = wandb.Api(timeout=100)
        runs = api.runs("%s/%s" % (wandb_name, project_name))
        for run in runs:
            if run.config["seed"] == seed and run.config["dataset"] == dataset_name and \
                    embed_model_config in run.config["embed_model_config"] and \
                    classifier_model_config in run.config["classifier_model_config"] and \
                    strategy_config in run.config["strategy_config"] and \
                    trainer_config in run.config["trainer_config"] and \
                    (("corrupter_config" not in run.config) and ("noiseless" in corrupter_config) or \
                     (corrupter_config in run.config["corrupter_config"])):
                labeled_idxs = {}
                for row in run.scan_history(keys=["i", "Label index"]):
                    labeled_idxs[row["i"]] = row["Label index"]
                idxs = np.zeros(len(labeled_idxs), dtype=int)
                for i in range(len(labeled_idxs)):
                    idxs[i] = labeled_idxs[i]
                return idxs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Random seed number to be used.")
    parser.add_argument("--wandb_name", type=str, help="Wandb user name.")
    parser.add_argument("--wandb_project", type=str, help="Wandb project with the active learning run.")
    parser.add_argument("--dataset", type=str, help="Dataset name.")
    parser.add_argument("--data_dir", type=str, help="Directory to store dataset.", default="./data")
    parser.add_argument("--embed_model_config", type=str, help="Path to the encoder model configuration file.",
                        default="none.json")
    parser.add_argument("--classifier_model_config", type=str, help="Path to model configuration file.")
    parser.add_argument("--strategy_config", type=str, help="Path to AL strategy configuration file.")
    parser.add_argument("--trainer_config", type=str, help="Path to trainer configuration file.")
    parser.add_argument("--corrupter_config", type=str, help="Path to corrupter configuration file.",
                        default="noiseless.json")
    parser.add_argument("--eval_metric", type=str, help="Metric name for evaluation")
    parser.add_argument("--eval_batch_size", type=int,
                        help="We train neural network after collecting batch_size number of new labels.")
    parser.add_argument("--eval_num_batch", type=int, help="Number of batches of collected labels.")
    parser.add_argument("--eval_classifier_model_config", type=str,
                        help="Path to model configuration file for evalutation.")
    parser.add_argument("--eval_trainer_config", type=str, help="Path to trainer configuration file for evaluation.")
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed + 42)

    wandb_name = args.wandb_name
    project_name = args.wandb_project
    dataset_name = args.dataset
    data_dir = args.data_dir
    metric_name = args.eval_metric
    batch_size = args.eval_batch_size
    num_batch = args.eval_num_batch

    with open(os.path.join("./configs/model", args.eval_classifier_model_config), "r") as f:
        classifier_model_config = json.load(f)
    with open("./configs/embed_model/none.json", "r") as f:
        embed_model_config = json.load(f)
        embed_model_config["num_output"] = 0
    with open(os.path.join("./configs/trainer", args.eval_trainer_config), "r") as f:
        trainer_config = json.load(f)
    with open(os.path.join("./configs/corrupter", args.corrupter_config), "r") as f:
        corrupter_config = json.load(f)

    run_name = "%s, proxy_model = %s, classifier_model=%s" % (
        args.strategy_config, args.classifier_model_config, classifier_model_config["model_name"])
    wandb.init(project="Point Evaluation, %s" % dataset_name, entity=wandb_name, name=run_name, config=vars(args))

    label_idxs = retrieve_run(seed, wandb_name, project_name, dataset_name, args.embed_model_config,
                              args.classifier_model_config, args.strategy_config, args.trainer_config, corrupter_config)
    assert len(label_idxs) >= num_batch * batch_size, "Insufficient number of labeled examples."

    # Use file system for data loading of large datasets.
    if "fs" in trainer_config and trainer_config["fs"]:
        torch.multiprocessing.set_sharing_strategy('file_system')

    # Retrieve ALDataset and number of classes.
    dataset = get_dataset(dataset_name, args.data_dir)
    classifier_model_config["num_output"] = dataset.get_num_classes()

    # Get corrupter setting.
    corrupter = get_corrupter_fn(corrupter_config["name"], corrupter_config)
    dataset.set_corrupter(corrupter)

    feature_extractor = None

    # Retrieve metric object.
    metric = get_metric(metric_name)

    # Retrieve model_fn for classifier.
    model_fn = get_model_fn(classifier_model_config["model_name"])

    # Construct trainer.
    trainer_config = get_fns(trainer_config)
    trainer_config = get_optimizer_fn(trainer_config)
    trainer_config = get_scheduler_fn(trainer_config)
    trainer = get_trainer(trainer_config["trainer_name"], trainer_config, dataset, model_fn, classifier_model_config,
                          metric, feature_extractor)

    # Active learning loop.
    for idx in range(1, num_batch + 1):
        new_idxs = label_idxs[(idx - 1) * batch_size: idx * batch_size]
        dataset.update_labeled_idxs(new_idxs)
        model = trainer.train()
        trainer.evaluate_on_train(model)
        trainer.evaluate_on_val(model)
        trainer.evaluate_on_test(model)
        metric_dict = trainer.compute_metric(idx)
        wandb.log(metric_dict)

    wandb.finish()
