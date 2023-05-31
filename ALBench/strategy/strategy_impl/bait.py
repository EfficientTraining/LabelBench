from copy import deepcopy
import gc
import time
from tqdm import tqdm
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from functorch import vmap
from ALBench.skeleton.active_learning_skeleton import Strategy, ALInput


def compute_single_hessian(x, p, ret_embed=False):
    x, p = np.expand_dims(x, 1), np.expand_dims(p, 1)  # x_single.shape = [embed_dim,1], p_single.shape = [n_classes,1]

    prob_matrix = np.diag(p) - np.outer(p, p)

    use_regression_not_classification = False

    if use_regression_not_classification:
        if ret_embed:
            return x
        else:
            return np.outer(x, x)
    else:
        if ret_embed:
            return np.kron(scipy.linalg.sqrtm(prob_matrix), x)
        else:
            return np.kron(prob_matrix, np.outer(x, x))


def woodbury(A_inv, U, sign):
    big_dim = A_inv.shape[0]
    small_dim = min(U.shape)

    if U.shape != (big_dim, small_dim):
        U = U.T

    outside = A_inv @ U
    inside = np.eye(small_dim) + sign * (U.T @ A_inv) @ U
    return A_inv - sign * outside @ np.linalg.inv(inside) @ outside.T


class BAIT(Strategy):
    strategy_name = "bait"

    def __init__(self, strategy_config, dataset):
        super(BAIT, self).__init__(strategy_config, dataset)
        self.input_types = {ALInput.TRAIN_PRED, ALInput.TRAIN_EMBEDDING}

        self.pca_dim = strategy_config["pca_dimension"] if "pca_dimension" in strategy_config else None
        self.num_proposed_adds = \
            strategy_config["num_proposed_additions"] if "num_proposed_additions" in strategy_config else 10
        self.num_complete_swaps = \
            strategy_config["num_complete_swaps"] if "num_complete_swaps" in strategy_config else 1

        self.curInv = 0  # Current inverse of the fisher information matrix on the selected samples.
        self.fisher_unlabeled = None  # Fisher information matrix on the unlabeled set.

        # The whole algorithm is trying to minimize the following optimization problem with fixed selection budget.
        # trace((self.curInv)@ self.fisher_unlabeled)

    def select(self, trainer, budget):
        preds, embs = trainer.retrieve_inputs(self.input_types)

        # PCA
        embs = embs - np.mean(embs, axis=0)

        if self.pca_dim is not None:
            cov = embs.T @ embs / embs.shape[0]
            v, w = np.linalg.eigh(cov)
            idx = (v.argsort()[::-1])[:self.pca_dim]  # Sort descending and get top

            embs = embs @ w[:, idx]

        labeled_set = set(list(self.dataset.labeled_idxs()))
        all_set = set(list(range(len(self.dataset))))
        unlabeled = list(all_set - labeled_set)

        self.fisher_total = self.compute_fisher(range(len(self.dataset)), preds, embs)
        self.fisher_total_sqrt = scipy.linalg.sqrtm(self.fisher_total)
        self.fisher_total_invsqrt = np.linalg.inv(self.fisher_total_sqrt)

        proposed_batch = list(np.random.choice(unlabeled, size=(budget,), replace=False))

        self.inv_fisher_proposed_adj = self.fisher_total_sqrt @ np.linalg.inv(
            self.compute_fisher(proposed_batch, preds, embs)) @ self.fisher_total_sqrt
        self.cur_val = np.trace(self.inv_fisher_proposed_adj)

        for _ in range(self.num_complete_swaps):
            for _ in tqdm(range(budget), desc="Swapping"):
                swap_remove = proposed_batch[0]

                swap_adds = np.random.choice(list(set(unlabeled) - set(proposed_batch)), size=(self.num_proposed_adds,),
                                             replace=False)

                remove_U = self.fisher_total_invsqrt @ compute_single_hessian(embs[swap_remove], preds[swap_remove],
                                                                              ret_embed=True)
                inter_inv_fisher_adj = woodbury(self.inv_fisher_proposed_adj, remove_U, sign=-1)

                best_swap_add = swap_remove
                best_swap_add_val = self.cur_val
                best_swap_matrix = self.inv_fisher_proposed_adj
                for swap_add in swap_adds:
                    add_U = self.fisher_total_invsqrt @ compute_single_hessian(embs[swap_add], preds[swap_add],
                                                                               ret_embed=True)

                    swap_inv_fisher_adj = woodbury(inter_inv_fisher_adj, add_U, sign=1)

                    swap_val = np.trace(swap_inv_fisher_adj)

                    if swap_val < best_swap_add_val:
                        best_swap_add = swap_add
                        best_swap_add_val = swap_val
                        best_swap_matrix = swap_inv_fisher_adj

                proposed_batch.append(best_swap_add)
                del proposed_batch[0]

                self.cur_val = best_swap_add_val
                self.inv_fisher_proposed_adj = best_swap_matrix

            self.inv_fisher_proposed_adj = self.fisher_total_sqrt @ \
                                           np.linalg.inv(self.compute_fisher(proposed_batch, preds, embs)) @ \
                                           self.fisher_total_sqrt
            self.cur_val = np.trace(self.inv_fisher_proposed_adj)

        return proposed_batch

    def compute_fisher(self, indices, probs, embs):
        embs = embs[indices]
        probs = probs[indices]
        total_fisher = 0
        for i in tqdm(range(len(indices)), desc="Computing Fisher Information"):
            total_fisher = total_fisher + compute_single_hessian(embs[i], probs[i])
        return total_fisher
