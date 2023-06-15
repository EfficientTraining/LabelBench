import numpy as np
from tqdm import tqdm
from scipy import stats

from LabelBench.skeleton.active_learning_skeleton import Strategy, ALInput


def distance(X1, X2, mu):
    Y1, Y2 = mu
    X1_vec, X1_norm_square = X1
    X2_vec, X2_norm_square = X2
    Y1_vec, Y1_norm_square = Y1
    Y2_vec, Y2_norm_square = Y2
    dist = X1_norm_square * X2_norm_square + Y1_norm_square * Y2_norm_square - 2 * (X1_vec @ Y1_vec) * (X2_vec @ Y2_vec)
    # Numerical errors may cause the distance squared to be negative.
    assert np.min(dist) / np.max(dist) > -1e-4
    dist = np.sqrt(np.clip(dist, a_min=0, a_max=None))
    return dist


def init_centers(X1, X2, chosen, mu, D2):
    if len(chosen) == 0:
        ind = np.argmax(X1[1] * X2[1])
        mu = [((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind]))]
        chosen = [ind]
        D2 = distance(X1, X2, mu[0]).ravel().astype(float)
        D2[np.array(chosen)] = 0
    else:
        newD = distance(X1, X2, mu[-1]).ravel().astype(float)
        D2 = np.minimum(D2, newD)
        D2[np.array(chosen)] = 0
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(Ddist)), Ddist))
        ind = customDist.rvs(size=1)[0]
        chosen_set = set(chosen)
        assert ind not in chosen_set, "%d, %s" % (ind, str(chosen_set))
        mu.append(((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind])))
        chosen.append(ind)
    return chosen, mu, D2


class BADGESampling(Strategy):
    strategy_name = "badge"

    def __init__(self, strategy_config, dataset):
        super(BADGESampling, self).__init__(strategy_config, dataset)
        self.input_types = {ALInput.TRAIN_PRED, ALInput.TRAIN_EMBEDDING}

    def select(self, trainer, budget):
        preds, embs = trainer.retrieve_inputs(self.input_types)
        unlabeled = self.dataset.unlabeled_idxs()
        unlabeled_set = set(unlabeled)
        chosen = []
        n = len(self.dataset)
        mu = None
        D2 = None
        embs = embs[unlabeled]
        emb_norms_square = np.sum(embs ** 2, axis=-1)
        max_inds = np.argmax(preds, axis=-1)
        # `probs` is (\one{y=i} - p).
        probs = -1 * preds
        probs[np.arange(n), max_inds] += 1
        probs = probs[unlabeled]
        prob_norms_square = np.sum(probs ** 2, axis=-1)
        for _ in tqdm(range(budget)):
            chosen, mu, D2 = init_centers((probs, prob_norms_square), (embs, emb_norms_square), chosen, mu, D2)
            unlabeled_set.remove(unlabeled[chosen[-1]])
        query_idxs = [unlabeled[i] for i in chosen]
        return query_idxs