from ALBench.skeleton.trainer_skeleton import trainers
from ALBench.skeleton.metric_skeleton import metrics
from ALBench.skeleton.model_skeleton import model_fns
from ALBench.skeleton.dataset_skeleton import datasets
from ALBench.skeleton.active_learning_skeleton import strategies
from ALBench.trainer.trainer import get_trainer
from ALBench.metric.metrics import get_metric
from ALBench.model.model import get_model_fn
from ALBench.dataset.datasets import get_dataset
from ALBench.strategy.strategies import get_strategy


if __name__ == "__main__":
    print(trainers)
    print(get_trainer("passive", None, None, None, None, None))

    print(metrics)
    print(get_metric("multi_class"))

    print(model_fns)
    print(get_model_fn("resnet18"))

    print(datasets)
    print(get_dataset("cifar10_imb_3", "./data"))

    print(strategies)
    print(get_strategy("random", None, None))
