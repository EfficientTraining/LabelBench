import torch
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10
from src.skeleton.dataset_skeleton import DatasetOnMemory, register_dataset, LabelType


@register_dataset("cifar10_imb", LabelType.MULTI_CLASS)
def get_cifar10_imb_dataset(n_class, *args):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    target_transform = transforms.Compose(
        [lambda x: torch.LongTensor([x]),
         lambda x: torch.flatten(F.one_hot(torch.clip(x, min=None, max=n_class - 1), n_class))])
    train_dataset = CIFAR10("./data", train=True, download=True, transform=transforms.ToTensor(), target_transform=target_transform)
    test_dataset = CIFAR10("./data", train=False, download=True, transform=transforms.ToTensor(), target_transform=target_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=40)
    train_imgs, train_labels = next(iter(train_loader))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=40)
    test_imgs, test_labels = next(iter(test_loader))
    train_dataset = DatasetOnMemory(train_imgs, train_labels, n_class, transform=transform)
    test_dataset = DatasetOnMemory(test_imgs, test_labels, n_class, transform=transform)

    rnd = np.random.RandomState(42)
    idxs = rnd.permutation(len(test_dataset))
    val_idxs, test_idxs = idxs[:-len(idxs) // 2], idxs[-len(idxs) // 2:]

    return train_dataset, Subset(test_dataset, val_idxs), Subset(test_dataset, test_idxs), train_labels, \
           test_labels[val_idxs], test_labels[test_idxs], n_class


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train, val, test, train_labels, val_labels, test_labels, _ = get_cifar10_imb_dataset(3)
    print(len(train), len(val), len(test), train_labels.shape, val_labels.shape, test_labels.shape)
    loader = DataLoader(train, batch_size=2)
    x, y = next(iter(loader))
    print(x.size(), y.size())

