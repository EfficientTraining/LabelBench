import numpy as np
from src.strategy import get_subprocedure
from src.strategy import MetaSamplingProcedure


class ThompsonMetaProcedure(MetaSamplingProcedure):
    def __init__(self, sub_names, obj):
        super().__init__(sub_names)
        self.obj = obj
        self.stats = None
        self.last_w = None

    def sample(self, trainer, model, embs, preds, labels, labeled, dataset, batch_size):
        sub_ps = get_subprocedure(self.sub_names, trainer, model, embs, preds, labels, labeled, dataset, batch_size)
        if self.stats is None:
            self.stats = np.ones((2, len(sub_ps), trainer.n_class))
        else:
            assert self.stats.shape[1] == len(sub_ps)
            assert self.stats.shape[2] == trainer.n_class
        v = self.get_v(labels, labeled, trainer.metric)

        algs = []
        new_label = []
        labeled_set = set(list(labeled))
        new_stats = np.zeros_like(self.stats)
        for t in range(batch_size):
            if trainer.multi_label_flag:
                theta = np.random.beta(self.stats[0, :, :], self.stats[1, :, :])
            else:
                theta = np.array([np.random.dirichlet(self.stats[0, i, :] + 1) for i in range(len(sub_ps))])
            expected_reward = theta @ v
            idx = np.argmax(expected_reward)
            algs.append(idx)
            new_label.append(sub_ps[idx].sample(labeled_set))
            labeled_set.add(new_label[-1])
            for p in sub_ps:
                p.update(new_label[-1])

            # Update stats in new_stats.
            label = labels[new_label[-1]]
            new_stats[0, idx] += label
            new_stats[1, idx] += 1 - label
        # As if labels are only revealed now.
        self.stats = self.stats * .9 + new_stats
        self.log(algs, len(sub_ps), sub_ps)
        return np.array(new_label)

    def get_v(self, labels, labeled, metric):
        if self.obj == "pos":
            return np.ones(labels.shape[1])
        elif self.obj == "div":
            w = np.sum(labels[labeled], axis=0)
            w = np.clip(w, a_min=1, a_max=None)
            w[(w / float(len(labeled))) > .5] *= -1
            return 1 / w
        elif self.obj == "acc":
            w = np.sum(labels[labeled], axis=0)
            w = np.clip(w, a_min=1, a_max=None)
            w[(w / float(len(labeled))) > .5] *= -1
            return metric.val_acc / w
