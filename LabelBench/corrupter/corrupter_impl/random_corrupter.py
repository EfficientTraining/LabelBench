import numpy as np
from LabelBench.skeleton.corrupter_skeleton import register_corrupter


@register_corrupter("random")
def noiseless_corrupter(corrupter_config, labels):
    n_examples = labels.shape[0]
    n_class = labels.shape[1]
    size = int(n_examples * corrupter_config["p"])
    labels = np.array(labels)
    label_idxs = np.argmax(labels, axis=-1)

    rnd = np.random.RandomState(321)
    idxs = rnd.choice(len(label_idxs), size=size, replace=False)
    for idx in idxs:
        cls = list(range(n_class))
        cls.remove(label_idxs[idx])
        labels[idx, label_idxs[idx]] = 0
        labels[idx, rnd.choice(cls)] = 1

    return labels
