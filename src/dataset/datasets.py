import src.dataset.dataset_impl
from src.skeleton.dataset_skeleton import datasets, ALDataset


def get_dataset(name, batch_size):
    if "_imb_" in name:
        n_class = int(name.split("_")[-1])
        name = name[:-(len(name.split("_")[-1]) + 1)]
    else:
        n_class = None
    data_type, get_fn = datasets[name]
    train_dataset, val_dataset, test_dataset, train_labels, val_labels, test_labels, num_classes = \
        get_fn(n_class, batch_size)
    return ALDataset(train_dataset, val_dataset, test_dataset, train_labels, val_labels, test_labels, data_type,
                     num_classes)


if __name__ == "__main__":
    print(get_dataset("cifar10_imb_3", 100))
