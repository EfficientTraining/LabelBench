import requests
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import transforms
from ALBench.skeleton.dataset_skeleton import DatasetOnMemory, register_dataset, LabelType, TransformDataset


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


@register_dataset("kuzushiji49", LabelType.MULTI_CLASS)
def get_kuzushiji49_dataset(data_dir, *args):
    n_class = 49

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])
    target_transform = transforms.Compose(
        [lambda x: torch.LongTensor([x]), lambda x: torch.flatten(F.one_hot(x, n_class))])
    train_dataset = Kuzushiji49(data_dir=data_dir, train=True, transform=transforms.ToTensor(),
                                target_transform=target_transform, download=True)
    test_dataset = Kuzushiji49(data_dir=data_dir, train=False, transform=transforms.ToTensor(),
                               target_transform=target_transform, download=True)

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
           test_labels[test_idxs], n_class, None
    #TODO: add class names


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train, val, test, train_labels, val_labels, test_labels, _, _ = get_kuzushiji49_dataset("./data")
    print(len(train), len(val), len(test), train_labels.shape, val_labels.shape, test_labels.shape)
    loader = DataLoader(train, batch_size=2)
    x, y = next(iter(loader))
    print(x.size(), y.size())
    print(y)
