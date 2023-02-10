from src.strategy import SamplingSubProcedure
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy import stats
import pdb


def init_centers(X, unlabeled, chosen, mu, D2):
    if len(chosen) == 0:
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        chosen = [ind]
    chosen_set = set(chosen)
    if len(mu) == 1:
        D2 = pairwise_distances(X, mu).ravel().astype(float)
        D2_unlabel = D2[unlabeled]
    else:
        newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)[unlabeled]
        D2_unlabel = np.minimum(D2[unlabeled], newD)
        D2[unlabeled] = D2_unlabel
    print(str(len(mu)) + '\t' + str(sum(D2_unlabel)), flush=True)
    if sum(D2_unlabel) == 0.0: pdb.set_trace()
    D2_unlabel = D2_unlabel.ravel().astype(float)
    Ddist = (D2_unlabel ** 2) / sum(D2_unlabel ** 2)
    customDist = stats.rv_discrete(name='custm', values=(np.arange(len(unlabeled)), Ddist))
    ind = unlabeled[customDist.rvs(size=1)[0]]
    assert ind not in chosen_set, "%d, %s" % (ind, str(chosen_set))
    mu.append(X[ind])
    chosen.append(ind)
    del D2_unlabel
    return chosen, mu, D2


class BADGESubProcedure(SamplingSubProcedure):
    def __init__(self, trainer, model, embs, preds, labels, labeled, dataset, batch_size, grad_embeddings, unlabeled):
        super(BADGESubProcedure, self).__init__(trainer, model, embs, preds, labels, labeled, dataset, batch_size)
        self.grad_embeddings = grad_embeddings
        self.preds = preds
        self.batch_size = batch_size
        self.chosen = []
        self.n = len(dataset)
        self.mu = None
        self.D2 = None
        assert grad_embeddings.shape[0] == len(unlabeled)
        self.unlabeled_set = set(unlabeled)
        self.idx2i = {}
        for i, idx in enumerate(unlabeled):
            self.idx2i[idx] = i
        self.unlabled_idx = unlabeled

    def sample(self, labeled_set):
        unlabeled_set = self.unlabeled_set - labeled_set
        i_lst = np.array([self.idx2i[idx] for idx in unlabeled_set])
        self.chosen, self.mu, self.D2 = init_centers(self.grad_embeddings, i_lst, self.chosen, self.mu, self.D2)
        return self.unlabled_idx[self.chosen[-1]]

    def __str__(self):
        return "BADGE"
