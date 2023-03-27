import wandb
import argparse
import torch
import numpy as np
import json
import os

from ALBench.dataset.datasets import get_dataset
from ALBench.dataset.feature_extractor import update_embed_dataset
from ALBench.model.model import get_model_fn
from ALBench.metric.metrics import get_metric
from ALBench.trainer.trainer import get_trainer, get_fns, get_optimizer_fn
from ALBench.strategy.strategies import get_strategy

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
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed + 42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    run_name = "%s, embed_model = %s, classifier_model=%s" % (
    strategy_config["strategy_name"], embed_model_config["model_name"], classifier_model_config["model_name"])
    wandb.init(project="Active Learning, %s, Batch Size=%d" % (dataset_name, batch_size), entity=wandb_name,
               name=run_name, config=vars(args))

    # Retrieve ALDataset and number of classes.
    dataset = get_dataset(dataset_name, args.data_dir)
    classifier_model_config["num_output"] = dataset.get_num_classes()

    # Construct embedding model if needed.
    if embed_model_config["model_name"] != "none":
        model_fn = get_model_fn(embed_model_config["model_name"])
        folder_name = os.path.join(data_dir, dataset_name)
        os.makedirs(folder_name, exist_ok=True)
        file_name = "{}/{}_{}".format(folder_name, dataset_name, embed_model_config["model_name"])
        input_dim = update_embed_dataset(model_fn, dataset, file_name, embed_model_config)
    else:
        input_dim = None

    # Retrieve metric object.
    metric = get_metric(metric_name)

    # Construct model if not linear.
    model_fn = get_model_fn(classifier_model_config["model_name"])  # Embed model or training model.

    # Construct trainer.
    trainer_config = get_fns(trainer_config)
    trainer_config = get_optimizer_fn(trainer_config)
    trainer = get_trainer(trainer_config["trainer_name"], trainer_config, dataset, model_fn, classifier_model_config,
                          metric, input_dim)

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
