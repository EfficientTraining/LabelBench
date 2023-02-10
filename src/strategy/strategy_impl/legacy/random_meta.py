import numpy as np
from src.strategy import get_subprocedure
from src.strategy import MetaSamplingProcedure


class RandomMetaProcedure(MetaSamplingProcedure):
    def __init__(self, sub_names):
        super().__init__(sub_names)

    def sample(self, trainer, model, embs, preds, labels, labeled, dataset, batch_size):
        sub_ps = get_subprocedure(self.sub_names, trainer, model, embs, preds, labels, labeled, dataset, batch_size)
        algs = np.random.choice(np.arange(len(sub_ps)), batch_size)
        self.log(algs, len(sub_ps), sub_ps)
        new_label = []
        labeled_set = set(list(labeled))
        for idx in algs:
            new_label.append(sub_ps[idx].sample(labeled_set))
            labeled_set.add(new_label[-1])
            for p in sub_ps:
                p.update(new_label[-1])
        return np.array(new_label)
