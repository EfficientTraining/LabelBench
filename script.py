from src.skeleton.trainer_skeleton import trainers
from src.skeleton.metric_skeleton import metrics
from src.skeleton.model_skeleton import model_fns
from src.skeleton.dataset_skeleton import datasets
from src.skeleton.active_learning_skeleton import strategies
from src.trainer.trainer import get_trainer
from src.metric.metrics import get_metric
from src.model.model import get_model_fn
from src.dataset.datasets import get_dataset
from src.strategy.strategies import get_strategy


if __name__ == "__main__":
    print(trainers)
    print(get_trainer("passive", None, None, None, None, None))

    print(metrics)
    print(get_metric("multi_class"))

    print(model_fns)
    print(get_model_fn("resnet18"))

    print(datasets)
    print(get_dataset("cifar10_imb_3", 100))

    print(strategies)
    print(get_strategy("random", None, None))
