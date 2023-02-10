from src.strategy import SamplingSubProcedure
import numpy as np


class EMALSubProcedure(SamplingSubProcedure):
    def __init__(self, trainer, model, embs, preds, labels, labeled, dataset, batch_size):
        super(EMALSubProcedure, self).__init__(trainer, model, embs, preds, labels, labeled, dataset, batch_size)
        labeled_set = set(list(labeled))
        all_set = set(list(range(len(dataset))))
        unlabeled = np.array(list(all_set - labeled_set))
        uncertainty = np.sum(np.abs(preds[unlabeled] - .5), axis=-1)
        self.top_idxs = unlabeled[np.argsort(uncertainty)[:batch_size]]
        self.idx = 0

    def sample(self, labeled_set):
        while self.top_idxs[self.idx] in labeled_set:
            self.idx += 1
        self.idx += 1
        return self.top_idxs[self.idx - 1]

    def __str__(self):
        return "EMAL"
