import numpy as np

from ALBench.skeleton.active_learning_skeleton import Strategy, ALInput


class RandomSampling(Strategy):
    strategy_name = "random"

    def __init__(self, strategy_config, dataset):
        super(RandomSampling, self).__init__(strategy_config, dataset)
        self.input_types = set()

    def select(self, trainer, budget):
        labeled_set = set(list(self.dataset.labeled_idxs()))
        all_set = set(list(range(len(self.dataset))))
        unlabeled = np.array(list(all_set - labeled_set))
        return np.random.choice(unlabeled, budget, replace=False)
