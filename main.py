import argparse
import json
import os

import numpy as np
import torch
import wandb
from LabelBench.trainer.trainer import (get_fns, get_optimizer_fn, get_scheduler_fn, get_trainer)
from LabelBench.dataset.datasets import get_dataset
from LabelBench.dataset.feature_extractor import FeatureExtractor
from LabelBench.metric.metrics import get_metric
from LabelBench.model.model import get_model_fn
from LabelBench.strategy.strategies import get_strategy
from LabelBench.corrupter.corrupter import get_corrupter_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Random seed number to be used.")
    parser.add_argument("--wandb_name", type=str, help="Wandb user name.")
    parser.add_argument("--dataset", type=str, help="Dataset name.")
    parser.add_argument("--data_dir", type=str, help="Directory to store dataset.", default="./data")
    parser.add_argument("--metric", type=str, help="Metric name.")
    parser.add_argument("--batch_size", type=int,
                        help="We train neural network after collecting batch_size number of new labels.")
    parser.add_argument("--num_batch", type=int, help="Number of batches of collected labels.")
    parser.add_argument("--embed_model_config", type=str, help="Path to the encoder model configuration file.",
                        default="none.json")
    parser.add_argument("--classifier_model_config", type=str, help="Path to model configuration file.")
    parser.add_argument("--strategy_config", type=str, help="Path to AL strategy configuration file.")
    parser.add_argument("--trainer_config", type=str, help="Path to trainer configuration file.")
    parser.add_argument("--corrupter_config", type=str, help="Path to corrupter configuration file.",
                        default="noiseless.json")
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed + 42)

    wandb_name = args.wandb_name
    dataset_name = args.dataset
    metric_name = args.metric
    batch_size = args.batch_size
    data_dir = args.data_dir
    num_batch = args.num_batch

    with open(os.path.join("./configs/model", args.classifier_model_config), "r") as f:
        classifier_model_config = json.load(f)
    with open(os.path.join("./configs/embed_model", args.embed_model_config), "r") as f:
        embed_model_config = json.load(f)
        embed_model_config["num_output"] = 0
    with open(os.path.join("./configs/strategy", args.strategy_config), "r") as f:
        strategy_config = json.load(f)
    with open(os.path.join("./configs/trainer", args.trainer_config), "r") as f:
        trainer_config = json.load(f)
    with open(os.path.join("./configs/corrupter", args.corrupter_config), "r") as f:
        corrupter_config = json.load(f)

    run_name = "%s, embed_model = %s, classifier_model=%s" % (
        strategy_config["strategy_name"], embed_model_config["model_name"], classifier_model_config["model_name"])
    wandb.init(project="Active Learning, %s, Batch Size=%d" % (dataset_name, batch_size), entity=wandb_name,
               name=run_name, config=vars(args))

    # Use file system for data loading of large datasets.
    if "fs" in trainer_config and trainer_config["fs"]:
        torch.multiprocessing.set_sharing_strategy('file_system')

    # Retrieve ALDataset and number of classes.
    dataset = get_dataset(dataset_name, args.data_dir)
    classifier_model_config["num_output"] = dataset.get_num_classes()

    # Get corrupter setting.
    corrupter = get_corrupter_fn(corrupter_config["name"], corrupter_config)
    dataset.set_corrupter(corrupter)

    # Construct embedding model and feature_extractor to get embedding if needed.
    if embed_model_config["model_name"] != "none":
        embed_model_fn = get_model_fn(embed_model_config["model_name"])
        folder_name = os.path.join(data_dir, dataset_name)
        os.makedirs(folder_name, exist_ok=True)
        file_name = "{}/{}_{}".format(folder_name, dataset_name, embed_model_config["model_name"])
        classifier_model_config["input_dim"] = embed_model_fn(embed_model_config).get_embedding_dim()
        feature_extractor = FeatureExtractor(embed_model_fn, file_name, embed_model_config)
    else:
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

    # Retrieve active learning strategy.
    strategy = get_strategy(strategy_config["strategy_name"], strategy_config, dataset)

    # Active learning loop.
    for idx in range(1, num_batch + 1):
        if idx == 1:
            # Initial seed set use random sampling.
            new_idxs = np.random.choice(np.arange(len(dataset)), size=batch_size, replace=False)
        else:
            # Use active learning strategy to select for batch other than the first.
            new_idxs = strategy.select(trainer, batch_size)
        dataset.update_labeled_idxs(new_idxs)
        model = trainer.train()
        trainer.evaluate_on_train(model)
        trainer.evaluate_on_val(model)
        trainer.evaluate_on_test(model)
        metric_dict = trainer.compute_metric(idx)
        wandb.log(metric_dict)

    # Log labeled examples in sampling order.
    for i, idx in enumerate(dataset.labeled_idxs()):
        wandb.log({
            "i": i,
            "Label index": idx,
        })
    wandb.finish()
