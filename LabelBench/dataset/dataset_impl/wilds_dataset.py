import torch
import torch.nn.functional as F
import numpy as np
import wilds
from torch.utils.data import Subset
from torchvision import transforms

from LabelBench.skeleton.dataset_skeleton import DatasetOnMemory, register_dataset, LabelType, TransformDataset
from LabelBench.dataset.dataset_impl.label_name.classnames import get_classnames


def get_wilds_dataset(data_dir, dataset_name, *args):
    wilds_dataset = wilds.get_dataset(dataset=dataset_name, root_dir=data_dir, download=True)

    classnames = get_classnames(dataset_name)
    n_class = len(classnames)

    train_dataset = wilds_dataset.get_subset("train")
    val_dataset = wilds_dataset.get_subset("id_val")
    test_dataset = wilds_dataset.get_subset("id_test")

    print("Number of train, val, test points:")
    print(len(train_dataset), len(val_dataset), len(test_dataset))

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomCrop(224, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    target_transform = transforms.Compose(
        [lambda x: torch.LongTensor([x]),
         lambda x: torch.flatten(F.one_hot(torch.clip(x, min=None, max=n_class - 1), n_class))])

    return TransformDataset(train_dataset, transform=train_transform, target_transform=target_transform, ignore_metadata=True), \
           TransformDataset(val_dataset, transform=test_transform, target_transform=target_transform, ignore_metadata=True), \
           TransformDataset(test_dataset, transform=test_transform, target_transform=target_transform, ignore_metadata=True), None, None, None, n_class, classnames


@register_dataset("iwildcam", LabelType.MULTI_CLASS)
def get_iwildcam_dataset(data_dir, *args):
    return get_wilds_dataset(data_dir, "iwildcam", *args)


@register_dataset("fmow", LabelType.MULTI_CLASS)
def get_iwildcam_dataset(data_dir, *args):
    return get_wilds_dataset(data_dir, "fmow", *args)


if __name__ == "__main__":
    pass
