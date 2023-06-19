import numpy as np

from LabelBench.skeleton.active_learning_skeleton import Strategy, ALInput


class RandomSampling(Strategy):
    strategy_name = "random"

    def __init__(self, strategy_config, dataset):
        super(RandomSampling, self).__init__(strategy_config, dataset)
        self.input_types = []

    def select(self, trainer, budget):
        unlabeled = self.dataset.unlabeled_idxs()
        return np.random.choice(unlabeled, budget, replace=False)
