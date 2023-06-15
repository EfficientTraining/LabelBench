import torch
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR100
from LabelBench.skeleton.dataset_skeleton import DatasetOnMemory, LabelType, register_dataset, TransformDataset
from LabelBench.dataset.dataset_impl.label_name.classnames import get_classnames

@register_dataset("cifar100_imb", LabelType.MULTI_CLASS)
def get_cifar100_imb_dataset(n_class, data_dir, *args):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    target_transform = transforms.Compose(
        [lambda x: torch.LongTensor([x]),
         lambda x: torch.flatten(F.one_hot(torch.clip(x, min=None, max=n_class - 1), n_class))])

    train_dataset = CIFAR100(data_dir, train=True, download=True, target_transform=target_transform)
    test_dataset = CIFAR100(data_dir, train=False, download=True, target_transform=target_transform)

    rnd = np.random.RandomState(42)
    idxs = rnd.permutation(len(test_dataset))
    val_idxs, test_idxs = idxs[:-len(idxs) // 2], idxs[-len(idxs) // 2:]

    val_dataset, test_dataset = Subset(test_dataset, val_idxs), Subset(test_dataset, test_idxs)

    if n_class<100:
        classnames = get_classnames("cifar100")[:n_class]+ ("others",)
    else:
        classnames = get_classnames("cifar100")

    return TransformDataset(train_dataset, transform=train_transform), \
           TransformDataset(val_dataset, transform=test_transform), \
           TransformDataset(test_dataset, transform=test_transform), None, None, None, n_class, classnames


@register_dataset("cifar100", LabelType.MULTI_CLASS)
def get_cifar100_dataset(data_dir, *args):
    n_class = 100
    return get_cifar100_imb_dataset(n_class, data_dir, *args)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train, val, test, train_labels, val_labels, test_labels, _, _ = get_cifar100_imb_dataset(3, "./data")
    print(len(train), len(val), len(test), train_labels.shape, val_labels.shape, test_labels.shape)
    loader = DataLoader(train, batch_size=2)
    x, y = next(iter(loader))
    print(x.size(), y.size())
    print(y)

    train, val, test, train_labels, val_labels, test_labels, _, _ = get_cifar100_dataset("./data")
    print(len(train), len(val), len(test), train_labels.shape, val_labels.shape, test_labels.shape)
    loader = DataLoader(train, batch_size=2)
    x, y = next(iter(loader))
    print(x.size(), y.size())
    print(y)
