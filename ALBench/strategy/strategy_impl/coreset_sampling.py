import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked

from ALBench.skeleton.active_learning_skeleton import Strategy, ALInput


class CoresetSampling(Strategy):
    # adapted from:
    # https://github.com/JordanAsh/badge/blob/master/query_strategies/core_set.py
    strategy_name = "coreset"

    def __init__(self, strategy_config, dataset):
        super(CoresetSampling, self).__init__(strategy_config, dataset)
        self.input_types = {ALInput.TRAIN_EMBEDDING}
        self.working_memory_GB = strategy_config["working_memory_GB"]

    def furthest_first(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            calc_min_dist = lambda D, _: np.amin(D, axis=1)
            working_memory_MiB = self.working_memory_GB * 10 ** 3
            min_dist_iter = pairwise_distances_chunked(X, X_set,
                                                       reduce_func=calc_min_dist,
                                                       working_memory=working_memory_MiB)
            min_dist = np.zeros(X.shape[0])
            chunk_idx = 0
            for min_dist_chunk in min_dist_iter:
                chunk_size = len(min_dist_chunk)
                min_dist[chunk_idx: (chunk_idx + chunk_size)] = min_dist_chunk
                chunk_idx += chunk_size

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            min_dist = np.minimum(min_dist, dist_new_ctr[:, 0])

        return idxs
    
    def select(self, trainer, budget):
        embs = trainer.retrieve_inputs(self.input_types)[0]
        labeled = self.dataset.labeled_idxs()
        unlabeled = self.dataset.unlabeled_idxs()

        chosen = self.furthest_first(embs[unlabeled], embs[labeled], budget)
        return unlabeled[chosen]
