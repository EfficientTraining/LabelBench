import numpy as np

from LabelBench.skeleton.active_learning_skeleton import Strategy, ALInput


class MarginSampling(Strategy):
    strategy_name = "margin"

    def __init__(self, strategy_config, dataset):
        super(MarginSampling, self).__init__(strategy_config, dataset)
        self.input_types = [ALInput.TRAIN_PRED]

    def select(self, trainer, budget):
        preds = trainer.retrieve_inputs(self.input_types)[0]
        unlabeled = self.dataset.unlabeled_idxs()
        n_class = preds.shape[1]
        if n_class > 2:
            preds_unlabeled = preds[unlabeled]
            preds_sorted = -np.sort(-preds_unlabeled, axis=-1)
            margin = preds_sorted[:, 0] - preds_sorted[:, 1]
        else:
            # Not exactly margin, but with the same ranking information.
            margin = np.abs(preds[unlabeled, 0] - .5)
        top_idxs = unlabeled[np.argsort(margin)[:budget]]
        return top_idxs
