import requests
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


class Kuzushiji49:
    resources = [
        ("http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz"),
        ("http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz"),
        ("http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz"),
        ("http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz")]

    training_file_imgs = "k49-train-imgs.npz"
    training_file_labels = "k49-train-labels.npz"
    test_file_imgs = "k49-test-imgs.npz"
    test_file_labels = "k49-test-labels.npz"

    def __init__(self, data_dir, train=True, transform=None, target_transform=None, download=True):
        self.data_dir = data_dir

        if download:
            self.download(train)

        train_data_imgs = os.path.join(self.data_dir, self.training_file_imgs)
        train_data_imgs = np.load(train_data_imgs)
        train_data_imgs = train_data_imgs.f.arr_0

        train_data_labels = os.path.join(self.data_dir, self.training_file_labels)
        train_data_labels = np.load(train_data_labels)
        train_data_labels = train_data_labels.f.arr_0

        test_data_imgs = os.path.join(self.data_dir, self.test_file_imgs)
        test_data_imgs = np.load(test_data_imgs)
        test_data_imgs = test_data_imgs.f.arr_0

        test_data_labels = os.path.join(self.data_dir, self.test_file_labels)
        test_data_labels = np.load(test_data_labels)
        test_data_labels = test_data_labels.f.arr_0

        self.transform = transform
        self.target_transform = target_transform

        if train:
            perm = np.random.RandomState(42).permutation(len(train_data_imgs))
            self.data = train_data_imgs[perm]
            self.targets = train_data_labels[perm]
        else:
            perm = np.random.RandomState(21).permutation(len(test_data_imgs))
            self.data = test_data_imgs[perm]
            self.targets = test_data_labels[perm]

    def __len__(self):
        return len(self.data) // 10

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')  # mode='L' - (8-bit pixels, black and white)

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def download(self, train):
        # download the Kuzushiji-49 dataset if it doesn't exist
        if self._check_exists():
            if train:
                print('Train dataset already exists!')
            else:
                print('Test dataset already exists!')
            return

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        for url in self.resources:
            filename = url.rpartition('/')[2]
            print('Downloading: ', filename)
            myfile = requests.get(url, allow_redirects=True)
            open(os.path.join(self.data_dir, filename), 'wb').write(myfile.content)

        print('All files downloaded!')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.data_dir, self.training_file_imgs)) and
                os.path.exists(os.path.join(self.data_dir, self.training_file_labels)) and
                os.path.exists(os.path.join(self.data_dir, self.test_file_imgs)) and
                os.path.exists(os.path.join(self.data_dir, self.test_file_labels)))


def get_kuzushiji49_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Resize(32)
    ])
    target_transform = transforms.Compose([lambda x: torch.LongTensor([x]), lambda x: torch.flatten(F.one_hot(x, 49))])
    train_dataset = Kuzushiji49(data_dir="./data", train=True, transform=transform, target_transform=target_transform,
                                download=True)
    test_dataset = Kuzushiji49(data_dir="./data", train=False, transform=transform, target_transform=target_transform,
                               download=True)
    return train_dataset, test_dataset


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train, test = get_kuzushiji49_dataset()
    print(len(train), len(test))
    loader = DataLoader(train, batch_size=2)
    x, y = next(iter(loader))
    print(x.size(), y.size())
