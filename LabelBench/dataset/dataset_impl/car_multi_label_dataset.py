import urllib.request as urllib2
import tarfile
from scipy.io import loadmat
import numpy as np
import os
from torch.utils.data import Dataset, Subset
from PIL import Image
from torchvision import transforms
from LabelBench.skeleton.dataset_skeleton import register_dataset, LabelType, TransformDataset


def getting_data(url, path):
    data = urllib2.urlopen(url)
    tar_package = tarfile.open(fileobj=data, mode="r:gz")
    tar_package.extractall(path)
    tar_package.close()
    return print("Data extracted and saved.")


def getting_metadata(url, filename):
    """
    Downloading a metadata file from a specific url and save it to the disc.
    """
    labels = urllib2.urlopen(url)
    file = open(filename, "wb")
    file.write(labels.read())
    file.close()
    return print("Metadata downloaded and saved.")


class MetaParsing:
    """
    Class for parsing image and meta-data for the Stanford car dataset to create a custom dataset.
    path: The filepah to the metadata in .mat format.
    *args: Accepts dictionaries with self-created labels which will be extracted from the metadata (e.g. {0: "Audi", 1: "BMW", 3: "Other").
    year: Can be defined to create two classes (<=year and later).
    """

    def __init__(self, data_dir):
        if not os.path.exists(os.path.join(data_dir, "carimages")):
            getting_data("http://ai.stanford.edu/~jkrause/car196/car_ims.tgz", os.path.join(data_dir, "carimages"))
        if not os.path.exists(os.path.join(data_dir, "car_metadata.mat")):
            getting_metadata("http://ai.stanford.edu/~jkrause/car196/cars_annos.mat", os.path.join(data_dir, "car_metadata.mat"))
        self.mat = loadmat(os.path.join(data_dir, "car_metadata.mat"))
        self.year = 2009
        self.annotations = np.transpose(self.mat["annotations"])
        # Extracting the file name for each sample
        self.file_names = [annotation[0][0][0].split("/")[-1] for annotation in self.annotations]
        # Extracting the index of the label for each sample
        self.label_indices = [annotation[0][5][0][0] for annotation in self.annotations]
        # Extracting the car names as strings
        self.car_names = [x[0] for x in self.mat["class_names"][0]]
        # Create a list with car names instead of label indices for each sample
        self.translated_car_names = [self.car_names[x - 1] for x in self.label_indices]
        self.name2class = {"Audi": 0, "BMW": 1, "Chevrolet": 2, "Dodge": 3, "Ford": 4, "Convertible": 5, "Coupe": 6,
                           "SUV": 7, "Van": 8}

    def parsing(self):
        labels = np.zeros((len(self.translated_car_names), len(self.name2class) + 1))
        for i, x in enumerate(self.translated_car_names):
            for name in self.name2class:
                if name in x:
                    labels[i, self.name2class[name]] = 1
            if int(x.split(" ")[-1]) <= self.year:
                labels[i, len(self.name2class)] = 1
        return labels, self.file_names, list(self.name2class.values())


class CarDataset(Dataset):
    def __init__(self, data_dir):
        self.labels, self.file_names, self.classnames = MetaParsing(data_dir).parsing()
        assert len(self.labels) == len(self.file_names)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_loc = os.path.join(os.path.join(self.data_dir, "carimages/car_ims"), self.file_names[idx])
        image = Image.open(img_loc).convert('RGB')

        return image, self.labels[idx]


@register_dataset("car_multi_label", LabelType.MULTI_LABEL)
def get_car_multi_label_dataset(data_dir, *args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CarDataset(data_dir)
    rnd = np.random.RandomState(42)
    idxs = rnd.permutation(len(dataset))
    train_idxs, val_idxs, test_idxs = idxs[:len(dataset) - len(dataset) // 5], \
                                      idxs[len(dataset) - len(dataset) // 5: -len(dataset) // 10], \
                                      idxs[-len(dataset) // 10:]
    train_dataset, val_dataset, test_dataset = \
        Subset(dataset, train_idxs), Subset(dataset, val_idxs), Subset(dataset, test_idxs)
    return TransformDataset(train_dataset, transform=transform), TransformDataset(val_dataset, transform=transform), \
           TransformDataset(test_dataset, transform=transform), dataset.labels[train_idxs], dataset.labels[val_idxs], \
           dataset.labels[test_idxs], 10, dataset.classnames


if __name__ == "__main__":
    train, val, test, _, _, _, _, _ = get_car_multi_label_dataset(100)
    print(train, val, test)
    print(len(train), len(val), len(test))
