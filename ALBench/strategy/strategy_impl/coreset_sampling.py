import numpy as np
from sklearn.metrics import pairwise_distances

from ALBench.skeleton.active_learning_skeleton import Strategy, ALInput


class CoresetSampling(Strategy):
    # adapted from:
    # https://github.com/JordanAsh/badge/blob/master/query_strategies/core_set.py
    strategy_name = "coreset"

    def __init__(self, strategy_config, dataset):
        super(CoresetSampling, self).__init__(strategy_config, dataset)
        self.input_types = {ALInput.TRAIN_EMBEDDING}

    def furthest_first(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs
    
    def select(self, trainer, budget):
        embs = trainer.retrieve_inputs(self.input_types)[0]
        labeled = self.dataset.labeled_idxs()
        unlabeled = self.dataset.unlabeled_idxs()

        chosen = self.furthest_first(embs[unlabeled], embs[labeled], budget)
        return unlabeled[chosen]
