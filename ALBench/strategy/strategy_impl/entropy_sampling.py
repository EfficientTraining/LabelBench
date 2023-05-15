import numpy as np
import torch
import torch.nn.functional as F

from ALBench.skeleton.active_learning_skeleton import Strategy, ALInput


class EntropySampling(Strategy):
    strategy_name = "entropy"

    def __init__(self, strategy_config, dataset):
        super(EntropySampling, self).__init__(strategy_config, dataset)
        self.input_types = {ALInput.TRAIN_PRED}

    def select(self, trainer, budget):
        preds = trainer.retrieve_inputs(self.input_types)[0]
        unlabeled = self.dataset.unlabeled_idxs()
        n_class = preds.shape[1]
        if n_class > 2:
            preds_unlabeled = preds[unlabeled]
            # Clip probability scores for numerical stability.
            log_preds = np.log2(np.clip(preds_unlabeled, a_min=1e-10, a_max=None))
            # Compute negative entropy.
            entropy = np.sum(log_preds * preds_unlabeled, axis=-1)
        else:
            # Not exactly entropy, but with the same ranking information.
            entropy = np.abs(preds[unlabeled, 0] - .5)
        top_idxs = unlabeled[np.argsort(entropy)[:budget]]
        return top_idxs
