import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import Subset
from torchvision.datasets import SVHN
from ALBench.skeleton.dataset_skeleton import DatasetOnMemory, LabelType, register_dataset, TransformDataset


@register_dataset("svhn_imb", LabelType.MULTI_CLASS)
def get_svhn_imb_dataset(n_class, data_dir, *args):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    target_transform = transforms.Compose(
        [lambda x: torch.LongTensor([x]),
         lambda x: torch.flatten(F.one_hot(torch.clip(x, min=None, max=n_class - 1), n_class))])
    train_dataset = SVHN(data_dir, split="train", download=True, transform=transforms.ToTensor(),
                         target_transform=target_transform)
    test_dataset = SVHN(data_dir, split="test", download=True, transform=transforms.ToTensor(),
                        target_transform=target_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False,
                                               num_workers=40)
    train_imgs, train_labels = next(iter(train_loader))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=40)
    test_imgs, test_labels = next(iter(test_loader))

    train_dataset = DatasetOnMemory(train_imgs, train_labels, n_class)
    test_dataset = DatasetOnMemory(test_imgs, test_labels, n_class)

    rnd = np.random.RandomState(42)
    idxs = rnd.permutation(len(test_dataset))
    val_idxs, test_idxs = idxs[:-len(idxs) // 2], idxs[-len(idxs) // 2:]

    val_dataset, test_dataset = Subset(test_dataset, val_idxs), Subset(test_dataset, test_idxs)

    return TransformDataset(train_dataset, transform=transform), TransformDataset(val_dataset, transform=transform), \
           TransformDataset(test_dataset, transform=transform), train_labels, test_labels[val_idxs], \
           test_labels[test_idxs], n_class


@register_dataset("svhn", LabelType.MULTI_CLASS)
def get_svhn_dataset(data_dir, *args):
    n_class = 10
    return get_svhn_imb_dataset(n_class, data_dir, *args)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train, val, test, train_labels, val_labels, test_labels, _ = get_svhn_imb_dataset(3,"./data")
    print(len(train), len(val), len(test), train_labels.shape, val_labels.shape, test_labels.shape)
    loader = DataLoader(train, batch_size=2)
    x, y = next(iter(loader))
    print(x.size(), y.size())
    print(y)

    train, val, test, train_labels, val_labels, test_labels, _ = get_svhn_dataset("./data")
    print(len(train), len(val), len(test), train_labels.shape, val_labels.shape, test_labels.shape)
    loader = DataLoader(train, batch_size=2)
    x, y = next(iter(loader))
    print(x.size(), y.size())
    print(y)
