import torch
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR100
from ALBench.skeleton.dataset_skeleton import DatasetOnMemory, LabelType, register_dataset


@register_dataset("cifar100_imb", LabelType.MULTI_CLASS)
def get_cifar100_imb_dataset(n_class, data_dir, *args):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]) 
    test_transform= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])  
    target_transform = transforms.Compose(
        [lambda x: torch.LongTensor([x]),
         lambda x: torch.flatten(F.one_hot(torch.clip(x, min=None, max=n_class - 1), n_class))])
    train_dataset = CIFAR100(data_dir, train=True, download=True, transform=transforms.ToTensor(),
                             target_transform=target_transform)
    test_dataset = CIFAR100(data_dir, train=False, download=True, transform=transforms.ToTensor(),
                            target_transform=target_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=40)
    train_imgs, train_labels = next(iter(train_loader))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=40)
    test_imgs, test_labels = next(iter(test_loader))
    train_dataset = DatasetOnMemory(train_imgs, train_labels, n_class, transform=train_transform)
    test_dataset = DatasetOnMemory(test_imgs, test_labels, n_class, transform=test_dataset)


    rnd = np.random.RandomState(42)
    idxs = rnd.permutation(len(test_dataset))
    val_idxs, test_idxs = idxs[:-len(idxs) // 2], idxs[-len(idxs) // 2:]

    return train_dataset, Subset(test_dataset, val_idxs), Subset(test_dataset, test_idxs), train_labels, \
           test_labels[val_idxs], test_labels[test_idxs], n_class

@register_dataset("cifar100", LabelType.MULTI_CLASS)
def get_cifar100_dataset(data_dir, *args):
    n_class = 100
    return get_cifar100_imb_dataset(n_class, data_dir, *args)

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train, val, test, train_labels, val_labels, test_labels, _ = get_cifar100_imb_dataset(3,"./data")
    print(len(train), len(val), len(test), train_labels.shape, val_labels.shape, test_labels.shape)
    loader = DataLoader(train, batch_size=2)
    x, y = next(iter(loader))
    print(x.size(), y.size())
    print(y)

    train, val, test, train_labels, val_labels, test_labels, _ = get_cifar100_dataset("./data")
    print(len(train), len(val), len(test), train_labels.shape, val_labels.shape, test_labels.shape)
    loader = DataLoader(train, batch_size=2)
    x, y = next(iter(loader))
    print(x.size(), y.size())
    print(y)