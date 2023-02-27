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


def test_dataset(dataset_name, data_dir = "./data"):

    if "_imb" in dataset_name:
        # if imbalance dataset, test with 3 classes
        dataset_name = f"{dataset_name}_3" 

    print(dataset_name)
    dataset = get_dataset(dataset_name, data_dir)
    train, val, test = dataset.get_input_datasets()
    print(len(train), len(val), len(test))
    x, y = train[0]
    print(x.size(), y.size())
    print(y)
    


if __name__ == "__main__":
    
    from torch.utils.data import DataLoader

    print(datasets)

    for dataset_name in datasets:
        test_dataset(dataset_name)
    
    
