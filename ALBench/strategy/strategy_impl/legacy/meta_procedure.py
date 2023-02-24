import wandb
import numpy as np


class MetaSamplingProcedure:
    def __init__(self, sub_names):
        self.sub_names = sub_names
        self.pulls = None

    def sample(self, trainer, model, embs, preds, labels, labeled, dataset, batch_size):
        raise NotImplementedError()

    def log(self, new_algs, num_algs, sub_ps):
        if self.pulls is None:
            self.pulls = np.zeros(num_algs)
        else:
            assert len(self.pulls) == num_algs
        for alg in new_algs:
            self.pulls[alg] += 1
        wandb.log({"Total Pulls": np.sum(self.pulls),
                   "Min Pulls": np.min(self.pulls),
                   "Max Pulls": np.max(self.pulls),
                   "Mean Pulls": np.mean(self.pulls),
                   "Median Pulls": np.median(self.pulls),
                   "Std Pulls": np.std(self.pulls),
                   })

        log_dict = {}
        for i, p in enumerate(sub_ps):
            log_dict[str(p)] = self.pulls[i]
        wandb.log(log_dict)
