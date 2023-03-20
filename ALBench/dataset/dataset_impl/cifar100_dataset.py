import torch
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR100
from ALBench.skeleton.dataset_skeleton import DatasetOnMemory, LabelType, register_dataset, TransformDataset

CLASSES = ("apple", "aquarium_fish", "baby", "bear", "beaver","bed","bee","beetle","bicycle","bottle","bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar","cattle","chair","chimpanzee","clock","cloud","cockroach","couch","cra","crocodile","cup","dinosaur","dolphin","elephant","flatfish","forest","fox","girl","hamster","house","kangaroo","keyboard","lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree","motorcycle","mountain","mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear","pickup_truck","pine_tree","plain","plate","poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket","rose","sea","seal","shark","shrew","skunk","skyscraper","snail","snake","spider","squirrel","streetcar","sunflower","sweet_pepper","table","tank","telephone","television","tiger","tractor","train","trout","tulip","turtle","wardrobe","whale","willow_tree","wolf","woman","worm")


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
        classnames = CLASSES[:n_class]+ ("others",)
    else:
        classnames = CLASSES

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
