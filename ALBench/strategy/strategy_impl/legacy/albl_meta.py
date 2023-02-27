import numpy as np
from src.strategy import get_subprocedure
from src.strategy import MetaSamplingProcedure


class ALBLMetaProcedure(MetaSamplingProcedure):
    def __init__(self, sub_names, delta=.1):
        super().__init__(sub_names)
        self.delta = delta
        self.start = True

    def sample(self, trainer, model, embs, preds, labels, labeled, dataset, batch_size):
        sub_ps = get_subprocedure(self.sub_names, trainer, model, embs, preds, labels, labeled, dataset, batch_size)
        if self.start:
            self.w = np.ones(len(sub_ps))
            self.pmin = 1.0 / (len(sub_ps) * 10.0)
            self.q = np.zeros(labels.shape[0])
            self.q[labeled] = 1.0
            self.start = False
        else:
            pred_label = (preds[self.last] > .5).astype(float)
            fn = (pred_label == labels[self.last]).astype(float)
            print(np.mean(fn))
            reward = np.mean(fn, axis=-1) / self.q[self.last]
            for i, alg in enumerate(self.last_algs):
                self.w[alg] *= np.exp(self.pmin / 2.0 * (
                            reward[i] + 1.0 / self.last_p[alg] * np.sqrt(np.log(len(sub_ps) / self.delta) / len(sub_ps))))
        self.w /= np.sum(self.w)
        p = (1.0 - len(sub_ps) * self.pmin) * self.w + self.pmin

        algs = np.random.choice(np.arange(len(sub_ps)), batch_size, p=p)
        self.log(algs, len(sub_ps), sub_ps)
        self.last_p = p
        self.last_algs = algs

        new_label = []
        labeled_set = set(list(labeled))
        for idx in algs:
            new_label.append(sub_ps[idx].sample(labeled_set))
            labeled_set.add(new_label[-1])
            for proc in sub_ps:
                proc.update(new_label[-1])

        new_label = np.array(new_label)
        self.q[new_label] = p[algs]
        self.last = np.array(new_label, copy=True)
        return new_label
