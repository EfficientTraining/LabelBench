from enum import Enum
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch


class LabelType(Enum):
    """Formats of label."""
    MULTI_CLASS = 1
    MULTI_LABEL = 2


datasets = {}


def register_dataset(name: str, type: LabelType):
    """
    Register dataset with dataset name and label type.
    :param str name: dataset name.
    :param LabelType type: the type of label for the dataset.
    :return: function decorator that registers the dataset.
    """

    def dataset_decor(get_fn):
        datasets[name] = (type, get_fn)
        return get_fn

    return dataset_decor


def get_labels(dataset):
    """
    Helper function to get all labels in a dataset with the original order.
    :param torch.utils.data.Dataset dataset: PyTorch dataset.
    :return: All labels in numpy array.
    """
    loader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=10, drop_last=False)
    labels = []
    for _, target in loader:
        labels.append(target)
    labels = torch.cat(labels, dim=0)
    return labels.numpy()


class DatasetOnMemory(Dataset):
    """
    A PyTorch dataset where all data lives on CPU memory.
    """

    def __init__(self, X, y, n_class):
        self.X = X
        self.y = y
        assert len(self.X) == len(self.y)
        self.n_class = n_class

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x = transforms.ToPILImage()(self.X[item])
        y = self.y[item]
        return x, y


class TransformDataset(Dataset):
    """
    A PyTorch Dataset where you can dynamically set transforms.

    Be careful about its behavior when combined with dataloaders!
    See https://discuss.pytorch.org/t/changing-transformation-applied-to-data-during-training/15671 for details.
    """

    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.__transform = transform
        self.__target_transform = target_transform
        self.__default_transform = transform
        self.__default_target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        if self.__transform:
            x = self.__transform(x)
        if self.__target_transform:
            y = self.__target_transform(y)
        return x, y

    def set_transform(self, transform):
        self.__transform = transform

    def set_target_transform(self, target_transform):
        self.__target_transform = target_transform

    def set_to_default_transform(self):
        self.__transform = self.__default_transform

    def set_to_default_target_transform(self):
        self.__target_transform = self.__default_target_transform


class ALDataset:
    """
    Dataset for active learning. The dataset contains all of training, validation and testing data as well as their
    embeddings. The dataset also tracks the examples that have been labeled.
    """

    def __init__(self, train_dataset, val_dataset, test_dataset, train_labels, val_labels, test_labels, label_type,
                 num_classes):
        """
        :param torch.utils.data.Dataset train_dataset: Training dataset that contains both examples and labels.
        :param torch.utils.data.Dataset val_dataset: Validation dataset that contains both examples and labels.
        :param torch.utils.data.Dataset test_dataset: Testing dataset that contains both examples and labels.
        :param Optional[numpy.ndarray] train_labels: All training labels for easy accessibility.
        :param Optional[numpy.ndarray] val_labels: All validation labels for easy accessibility.
        :param Optional[numpy.ndarray] test_labels: All testing labels for easy accessibility.
        :param LabelType label_type: Type of labels.
        :param int num_classes: Number of classes of the dataset.
        """
        assert isinstance(train_dataset, TransformDataset), "Training dataset must be a TransformDataset."
        assert isinstance(val_dataset, TransformDataset), "Validation dataset must be a TransformDataset."
        assert isinstance(test_dataset, TransformDataset), "Test dataset must be a TransformDataset."
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.label_type = label_type
        self.num_classes = num_classes
        self.train_emb = None
        self.val_emb = None
        self.test_emb = None
        self.__labeled_idxs = None
        self.train_labels = get_labels(train_dataset) if train_labels is None else train_labels
        self.val_labels = get_labels(val_dataset) if val_labels is None else val_labels
        self.test_labels = get_labels(train_dataset) if test_labels is None else test_labels

    def update_emb(self, train_emb, val_emb, test_emb):
        """
        Update with the latest feature embeddings.

        :param numpy.ndarray train_emb: Embeddings of training examples.
        :param numpy.ndarray val_emb: Embeddings of validation examples.
        :param numpy.ndarray test_emb: Embeddings of testing examples.
        :return:
        """
        self.train_emb = train_emb
        self.val_emb = val_emb
        self.test_emb = test_emb

    def update_labeled_idxs(self, new_idxs):
        """
        Insert the examples that have been newly labeled to update the dataset tracker.

        :param List new_idxs: list of newly labeled indexes.
        """
        if self.__labeled_idxs is None:
            self.__labeled_idxs = np.array(new_idxs)
        else:
            self.__labeled_idxs = np.concatenate((self.__labeled_idxs, np.array(new_idxs)))

    def get_embedding_datasets(self):
        """
        Construct PyTorch datasets of (embedding, label) pairs for all of training, validation and testing.
        :return: three PyTorch datasets for training, validation and testing respectively.
        """
        if self.train_emb is None or self.val_emb is None or self.test_emb is None:
            raise Exception("Embedding is not initialized.")
        return DatasetOnMemory(self.train_emb, self.train_labels, self.num_classes), \
               DatasetOnMemory(self.val_emb, self.val_labels, self.num_classes), \
               DatasetOnMemory(self.test_emb, self.test_labels, self.num_classes)

    def get_input_datasets(self):
        """
        Retrieves PyTorch datasets of (raw data, label) pairs for all of training, validation and testing.
        :return: three PyTorch datasets for training, validation and testing respectively.
        """
        return self.train_dataset, self.val_dataset, self.test_dataset

    def __len__(self):
        """Length of the training set."""
        return len(self.train_dataset)

    def num_labeled(self):
        """Number of labeled examples in the pool."""
        return len(self.__labeled_idxs)

    def labeled_idxs(self):
        """Indexes of the labeled examples in chronological order."""
        return np.array(self.__labeled_idxs)
