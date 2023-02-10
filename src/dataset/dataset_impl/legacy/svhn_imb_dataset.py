import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import SVHN
from src.skeleton.dataset_skeleton import DatasetOnMemory


def get_svhn_imb_dataset(n_class):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    target_transform = transforms.Compose(
        [lambda x: torch.LongTensor([x]),
         lambda x: torch.flatten(F.one_hot(torch.clip(x, min=None, max=n_class - 1), n_class))])
    train_dataset = SVHN("./data", split="train", download=True, transform=transform,
                         target_transform=target_transform)
    test_dataset = SVHN("./data", split="test", download=True, transform=transform,
                        target_transform=target_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False,
                                               num_workers=40)
    train_imgs, train_labels = next(iter(train_loader))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=40)
    test_imgs, test_labels = next(iter(test_loader))
    train_dataset = DatasetOnMemory(train_imgs, train_labels, n_class)
    test_dataset = DatasetOnMemory(test_imgs, test_labels, n_class)
    torch.utils.data.random_split(test, [batch_size // 2, len(test) - batch_size // 2],
                                  generator=torch.Generator().manual_seed(42))
    return train_dataset, test_dataset


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train, test = get_svhn_imb_dataset(3)
    print(len(train), len(test))
    loader = DataLoader(train, batch_size=2)
    x, y = next(iter(loader))
    print(x.size(), y.size())
