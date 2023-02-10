from src.dataset import get_dataset, get_labels
import numpy as np


if __name__ == "__main__":
    for name in ["car", "celeb", "voc", "coco"]:
        train, _, test, _, n_class = get_dataset(name, 100)
        labels = get_labels(train)
        n = float(labels.shape[0])
        pos = np.sum(labels, axis=0) / n
        assert len(pos) == n_class, "%d != %d" % (len(pos), n_class)
        print(name, len(train), len(test), np.min(pos) / np.max(pos), np.mean(pos))

    for name in ["cifar10_imb_2", "cifar100_imb_10", "svhn_imb_2", "caltech", "kuzushiji"]:
        train, _, test, _, n_class = get_dataset(name, 100)
        labels = get_labels(train)
        n = float(labels.shape[0])
        pos = np.sum(labels, axis=0) / n
        assert len(pos) == n_class, "%d != %d" % (len(pos), n_class)
        print(name, len(train), len(test), np.min(pos) / np.max(pos))
