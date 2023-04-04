import numpy as np

from ALBench.skeleton.active_learning_skeleton import Strategy, ALInput


class ConfidenceSampling(Strategy):
    strategy_name = "confidence"

    def __init__(self, strategy_config, dataset):
        super(ConfidenceSampling, self).__init__(strategy_config, dataset)
        self.input_types = {ALInput.TRAIN_PRED}

    def select(self, trainer, budget):
        preds = trainer.retrieve_inputs(self.input_types)[0]
        labeled_set = set(list(self.dataset.labeled_idxs()))
        all_set = set(list(range(len(self.dataset))))
        unlabeled = np.array(list(all_set - labeled_set))
        n_class = preds.shape[1]
        if n_class > 2:
            confidence = np.max(preds[unlabeled], axis=-1)
        else:
            confidence = np.abs(preds[unlabeled, 0] - .5)
        top_idxs = unlabeled[np.argsort(confidence)[:budget]]
        return top_idxs
