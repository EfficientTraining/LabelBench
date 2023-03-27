import ALBench.dataset.dataset_impl
from ALBench.skeleton.dataset_skeleton import datasets, ALDataset
import torch

def get_dataset(name, data_dir,lazy_mode= False):
    if "_imb_" in name:
        n_class = int(name.split("_")[-1])
        name = name[:-(len(name.split("_")[-1]) + 1)]
    else:
        n_class = None
    data_type, get_fn = datasets[name]
    if n_class is None:
        train_dataset, val_dataset, test_dataset, train_labels, val_labels, test_labels, num_classes, classnames = get_fn(data_dir)
    else:
        train_dataset, val_dataset, test_dataset, train_labels, val_labels, test_labels, num_classes, classnames = get_fn(n_class,
                                                                                                              data_dir)
    if lazy_mode == True:
        # When we don't need the to compute labels explicitly, e.g. when we use the dataset for end-to-end finetune, set to the lazy mode.
        train_labels = torch.empty(len(train_dataset))
        val_labels = torch.empty(len(train_dataset))
        test_labels = torch.empty(len(train_dataset))

    return ALDataset(train_dataset, val_dataset, test_dataset, train_labels, val_labels, test_labels, data_type,
                     num_classes, classnames)
