from src.strategy import SamplingSubProcedure
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch


class FinetuneDataset(Dataset):
    def __init__(self, dataset, labeled):
        self.dataset = dataset
        self.labeled = np.zeros(len(dataset))
        self.labeled[labeled] = 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        X, y = self.dataset.__getitem__(item)
        return X, y, self.labeled[item]


def weak_loss(pred, target, labeled):
    labeled = labeled.float().cuda().unsqueeze(-1)
    labeled_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * labeled
    unlabeled_loss = F.binary_cross_entropy_with_logits(pred, pred, reduction="none") * (1 - labeled)
    return torch.mean(labeled_loss + unlabeled_loss)


class WeakSupSubProcedure(SamplingSubProcedure):
    def __init__(self, trainer, model, embs, preds, labels, labeled, dataset, batch_size):
        super(WeakSupSubProcedure, self).__init__(trainer, model, embs, preds, labels, labeled, dataset, batch_size)
        labeled_set = set(list(labeled))
        all_set = set(list(range(len(dataset))))
        unlabeled = np.array(list(all_set - labeled_set))
        finetune_dataset = FinetuneDataset(dataset, labeled)
        model = trainer.train(finetune_dataset, None, finetune=(model, weak_loss, 2))
        preds, labels, losses = trainer.test(dataset, model)
        uncertainty = preds[unlabeled] * np.log2(preds[unlabeled]) + (1 - preds[unlabeled]) * np.log2(
            1 - preds[unlabeled])
        uncertainty = np.mean(uncertainty, axis=-1)
        self.top_idxs = unlabeled[np.argsort(uncertainty)[:batch_size]]
        self.idx = 0

    def sample(self, labeled_set):
        while self.top_idxs[self.idx] in labeled_set:
            self.idx += 1
        self.idx += 1
        return self.top_idxs[self.idx - 1]

    def __str__(self):
        return "Weak Sup"
