import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import LabelBench.dataset.dataset_impl
from LabelBench.skeleton.dataset_skeleton import ALDataset, datasets


def get_dataset(name, data_dir):
    if "_imb_" in name:
        n_class = int(name.split("_")[-1])
        name = name[:-(len(name.split("_")[-1]) + 1)]
    else:
        n_class = None
    data_type, get_fn = datasets[name]
    if n_class is None:
        train_dataset, val_dataset, test_dataset, train_labels, val_labels, test_labels, num_classes, classnames = \
            get_fn(data_dir)
    else:
        train_dataset, val_dataset, test_dataset, train_labels, val_labels, test_labels, num_classes, classnames = \
            get_fn(n_class, data_dir)

    # Use thunks for lazy evaluation.
    train_labels = lambda train_dataset=train_dataset, data_dir=data_dir, name=name, split="train": get_labels(
        train_dataset, data_dir, name, split)
    val_labels = lambda val_dataset=val_dataset, data_dir=data_dir, name=name, split="val": get_labels(
        val_dataset, data_dir, name, split)
    test_labels = lambda test_dataset=test_dataset, data_dir=data_dir, name=name, split="test": get_labels(
        test_dataset, data_dir, name, split)

    return ALDataset(train_dataset, val_dataset, test_dataset, train_labels, val_labels, test_labels, data_type,
                     num_classes, classnames)


def get_labels(dataset, data_dir, name, split):
    """
    Helper function to get all labels in a dataset with the original order.
    :param torch.utils.data.Dataset dataset: PyTorch dataset.
    :param str name: Name of dataset to be added the label file.
    :param str split: Dataset split. Takes value "train", "val" and "test".
    :return: All labels in numpy array.
    """
    folder_name = os.path.join(data_dir, "labels")
    os.makedirs(folder_name, exist_ok=True)

    if os.path.exists(f'{folder_name}_{name}_{split}.pt'):
        print(f"Loading labels from {folder_name}_{name}_{split}.pt")
        labels = torch.load(f'{folder_name}_{name}_{split}.pt')
    else:
        loader = DataLoader(dataset, batch_size=500, shuffle=False, num_workers=40, drop_last=False)
        labels = []
        print("Getting labels...")
        for _, target in tqdm(loader):
            labels.append(target)
        labels = torch.cat(labels, dim=0)
        torch.save(labels, f'{folder_name}_{name}_{split}.pt', pickle_protocol=4)

    return labels.numpy()
