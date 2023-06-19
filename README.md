# LabelBench: A Comprehensive Framework for Benchmarking Label-Efficient Learning
Welcome to LabelBench, where we evaluate label-efficient learning performance with a 
concerted combination of large pretrained models, semi-supervised learning and active
learning algorithms. We encourage researchers to contribute datasets, pretrained models,
semi-supervised training algorithms and active learning algorithms to this repo.
Additionally, results and findings can be reported under the `results` directory.

## Setup Python Environment
With conda:
```
conda create -n labelbench python=3.9
conda activate labelbench
pip install -r requirements.txt
```

With pip:
```
pip install -r requirements.txt
```

## Running LabelBench
LabelBench currently have the following two workflows.
### Selection of Examples for Oracle Annotation
As an example below, you can launch an experiment on CIFAR-10 while using FlexMatch to
fine-tune a pretrained CLIP model end-to-end. In addition, this experiment uses Margin
Sampling as the active learning selection strategy.
```
python main.py --seed 1234 --wandb_name $WANDB_NAME --dataset cifar10 --data_dir ./data --metric multi_class --batch_size 1000 --num_batch 10 --embed_model_config none.json --classifier_model_config clip_ViTB32_pretrained.json --strategy_config margin_sampling.json --trainer_config cifar10/passive/clip_ViTB32_finetune.json
```
Here `batch_size` specifies the number of examples to be annotated before everytime
the model is updated and `num_batch` specifies the total number of batch of labeled
examples/model updates.
For selection-via-proxy, specify the `embed_model_config` to one of the pretrained
models (see options under [configs](./configs/embed_model)) and `classifier_model_config` to
`linear.json` or `shallow.json`. This will allow the selection procedure to only retrain
a linear head or shallow neural network everytime a new batch of examples is labeled.

### Evaluation for Selection-Via-Proxy
Once the selection is finished using a proxy model such as linear model or shallow
networks, one can evaluate the end-to-end fine-tuning performance based on the
selected examples. As an example, the following retrieves the examples selected
by a proxy linear model and evaluate the end-to-end fine-tuning performance.
```
python point_evaluation.py --seed 1234 --wandb_name  $WANDB_NAME --wandb_project "Active Learning, cifar10, Batch Size=1000" --dataset cifar10 --data_dir ./data --embed_model_config clip_ViTB32 --classifier_model_config linear --strategy_config margin_sampling --trainer_config flexmatch --eval_metric multi_class --eval_batch_size 10000 --eval_num_batch 1 --eval_classifier_model_config clip_ViTB32_pretrained.json --eval_trainer_config cifar10/flexmatch/clip_ViTB32_finetune.json
```
To obtain end-to-end performance evaluation for every batch, simply set `eval_batch_size`
and `eval_num_batch` to the numbers used during the selection procedure.

### Parallel Experiment Running
To launch multiple processes in parallel from `main.py` and `point_evaluation.py`,
use `mp_launcher.py` and `mp_eval_launcer.py` respectively.
More example usages can be in [example_run.sh](./example_run.sh).

## File Structure
While the above section has introduced the entry points to our codebase, we now detail
the structure of the rest.
- `configs`: all of the configuration files for starting experiments. It also helps standardize the experiment settings.
- `LabelBench/skeleton`: abstract classes and general utility functions and classes that are useful throughout the entire codebase.
- `LabelBench/dataset`: data loading procedures for individual datasets and pre-computation of embeddings using pretrained models.
- `LabelBench/metrics`: compute metric logs for wandb based on model predictions.
- `LabelBench/model`: model classes and pretrained weights loading.
- `LabelBench/templates`: templates for loading zero-shot prediction heads.
- `LabelBench/strategy`: different active learning strategies for selection of unlabeled examples.
- `LabelBench/trainer`: training strategies when given a partially labeled dataset. Currently we include both supervised passive trainer and semi-supervised trainer such as FlexMatch.
- `results`: result fetching from WandB and plotting tools. Also keeps track of self-reported results to date.

## Getting Started and Contributing to LabelBench
From adding a new dataset or model to adding new algorithms of active or semi-supervised learning, you only need to work
with part of this repository. To start, it is important to understand the `skeleton` directory and the directory you
would like to contribute to, e.g. `dataset`, `metrics`, `model`, `strategy` or `trainer`.
To add your own implementation to any of these directories, we provide more details in their corresponding
`README.md` files.

### Result Fetching and Reporting
Under `results`, we provide python scripts to fetch experiment results from WandB to local computer.
Additionally plotting tools are also provided. For example usages, please see
[results/example_run.sh](./results/example_run.sh).

To report the results of your experiments, please submit a pull request with the fetched result files
along with any additional code and configuration files from your implementation so others can potentially
reproduce the results.
