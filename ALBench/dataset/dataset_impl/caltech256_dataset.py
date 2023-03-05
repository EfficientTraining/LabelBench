from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Subset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
import os
from PIL import Image
from ALBench.skeleton.dataset_skeleton import register_dataset, LabelType, TransformDataset


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech256(VisionDataset):
    """`Caltech 256 <https://data.caltech.edu/records/20087>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``caltech256`` exists or will be saved to if download is set to True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
            self,
            root,
            transform=None,
            target_transform=None,
            download: bool = False,
    ) -> None:
        super().__init__(os.path.join(root, "caltech256"), transform=transform, target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.categories = sorted(os.listdir(os.path.join(self.root, "256_ObjectCategories")))
        self.index = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(
                [
                    item
                    for item in os.listdir(os.path.join(self.root, "256_ObjectCategories", c))
                    if item.endswith(".jpg")
                ]
            )
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = pil_loader(
            os.path.join(
                self.root,
                "256_ObjectCategories",
                self.categories[self.y[index]],
                f"{self.y[index] + 1:03d}_{self.index[index]:04d}.jpg",
            )
        )

        target = self.y[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "256_ObjectCategories"))

    def __len__(self) -> int:
        return len(self.index)

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            "https://drive.google.com/file/d/1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK",
            self.root,
            filename="256_ObjectCategories.tar",
            md5="67b4f42ca05d46448c6bb8ecd2220f6d",
        )


@register_dataset("caltech256", LabelType.MULTI_CLASS)
def get_caltech256_dataset(data_dir, *args):
    transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose(
        [lambda x: torch.LongTensor([x]) % 256, lambda x: torch.flatten(F.one_hot(x, 256))])
    dataset = Caltech256(root=data_dir, target_transform=target_transform, download=True)
    rnd = np.random.RandomState(42)
    idxs = rnd.permutation(len(dataset))
    train_idxs, val_idxs, test_idxs = idxs[:len(dataset) - len(dataset) // 5], \
                                      idxs[len(dataset) - len(dataset) // 5: -len(dataset) // 10], \
                                      idxs[-len(dataset) // 10:]
    train_dataset, val_dataset, test_dataset = \
        Subset(dataset, train_idxs), Subset(dataset, val_idxs), Subset(dataset, test_idxs)
    return TransformDataset(train_dataset, transform=transform), TransformDataset(val_dataset, transform=transform), \
           TransformDataset(test_dataset, transform=transform), None, None, None, 256


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train, val, test, _, _, _, _ = get_caltech256_dataset()
    print(len(train), len(val), len(test))
    loader = DataLoader(train, batch_size=2)
    x, y = next(iter(loader))
    print(x.size(), y.size())
