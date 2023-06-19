from copy import deepcopy
import gc
import time
from tqdm import tqdm
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from functorch import vmap
from LabelBench.skeleton.active_learning_skeleton import Strategy, ALInput

def mysqrtm(A):
    lamda, V = np.linalg.eigh(A)

    assert(min(lamda) > -10**-6)

    lamda = np.maximum(0,lamda)

    return V @ np.diag(np.sqrt(lamda)) @ V.T



def woodbury(A_inv, U, sign):
    big_dim = A_inv.shape[0]
    small_dim = min(U.shape)

    assert(U.shape == (big_dim, small_dim))

    outside = A_inv @ U
    inside = sign * np.eye(small_dim) + (U.T @ A_inv) @ U

    return A_inv - outside @ np.linalg.inv(inside) @ outside.T


class BAIT(Strategy):
    strategy_name = "bait"

    def __init__(self, strategy_config, dataset):
        super(BAIT, self).__init__(strategy_config, dataset)
        self.input_types = [ALInput.TRAIN_PRED, ALInput.TRAIN_EMBEDDING]

        self.pca_dim = strategy_config["pca_dimension"] if "pca_dimension" in strategy_config else None
        self.num_proposed_adds = \
            strategy_config["num_proposed_additions"] if "num_proposed_additions" in strategy_config else 10
        self.num_complete_swaps = \
            strategy_config["num_complete_swaps"] if "num_complete_swaps" in strategy_config else 1

        self.curInv = 0  # Current inverse of the fisher information matrix on the selected samples.
        # The whole algorithm is trying to minimize the following optimization problem with fixed selection budget.
        # trace((self.curInv)@ self.fisher_unlabeled)

    def select(self, trainer, budget):
        preds, embs = trainer.retrieve_inputs(self.input_types)

        n_classes = preds.shape[1]
        self.orthogonal_transform, _ = np.linalg.qr(np.ones((n_classes,1)), mode='complete')

        # PCA
        embs = embs - np.mean(embs, axis=0)

        #evals = np.linalg.eigvalsh(embs.T @ embs)
        #print(evals)
        
        if self.pca_dim is not None:
            cov = embs.T @ embs / embs.shape[0]
            v, w = np.linalg.eigh(cov)
            idx = (v.argsort()[::-1])[:self.pca_dim]  # Sort descending and get top

            embs = embs @ w[:, idx]

        #evals = np.linalg.eigvalsh(embs.T @ embs)
        #print(evals)
      
        labeled_set = set(list(self.dataset.labeled_idxs()))
        all_set = set(list(range(len(self.dataset))))
        unlabeled = list(all_set - labeled_set)

        prev_labeled = list(self.dataset.labeled_idxs())

        fisher_total = self.compute_fisher(range(len(self.dataset)), preds, embs)


        #evals = np.linalg.eigvalsh(fisher_total)
        #print(evals)
        #print(np.sum(evals > 10**-6))

        print(fisher_total.shape)

        fisher_total_sqrt = mysqrtm(fisher_total)
        fisher_total_invsqrt = np.linalg.inv(fisher_total_sqrt)

        proposed_batch = list(np.random.choice(unlabeled, size=(budget,), replace=False))

        proposed_fisher_ratio = np.linalg.inv(0 * np.eye(fisher_total.shape[0]) + \
                fisher_total_invsqrt @ self.compute_fisher(prev_labeled + proposed_batch, preds, embs) @ fisher_total_invsqrt)
        self.cur_val = np.trace(proposed_fisher_ratio)
        print("Starting value: {}".format(self.cur_val))

        for _ in range(self.num_complete_swaps):
            for _ in tqdm(range(budget), desc="Swapping"):
                swap_remove = proposed_batch[0]

                swap_adds = list(np.random.choice(list(set(unlabeled) - set(proposed_batch)), size=(self.num_proposed_adds,),
                                             replace=False))

                remove_U = fisher_total_invsqrt @ self.compute_single_fisher(embs[swap_remove], preds[swap_remove],
                                                                              ret_embed=True)
                fisher_ratio_without = woodbury(proposed_fisher_ratio, remove_U, sign=-1)
                    
                best_swap_add = swap_remove
                best_swap_add_val = self.cur_val
                best_swap_fisher_ratio = proposed_fisher_ratio
                for swap_add in swap_adds:
                    add_U = fisher_total_invsqrt @ self.compute_single_fisher(embs[swap_add], preds[swap_add],
                                                                               ret_embed=True)

                    fisher_ratio_swap = woodbury(fisher_ratio_without, add_U, sign=1)
                        
                    swap_val = np.trace(fisher_ratio_swap)

                    if swap_val < best_swap_add_val:
                        best_swap_add = swap_add
                        best_swap_add_val = swap_val
                        best_swap_fisher_ratio = fisher_ratio_swap

                proposed_batch.append(best_swap_add)
                del proposed_batch[0]

                self.cur_val = best_swap_add_val
                proposed_fisher_ratio = best_swap_fisher_ratio
                
            print("Woodbury current value: {}".format(self.cur_val))
            proposed_fisher_ratio = np.linalg.inv(0 * np.eye(fisher_total.shape[0]) + \
                    fisher_total_invsqrt @ self.compute_fisher(prev_labeled + proposed_batch, preds, embs) @ fisher_total_invsqrt)
            self.cur_val = np.trace(proposed_fisher_ratio)
            print("Recomputed current value: {}".format(self.cur_val))

        return proposed_batch

    def compute_fisher(self, indices, preds, embs):
        total_fisher = 0
        for i in tqdm(indices, desc="Computing Fisher Information"):
            total_fisher = total_fisher + self.compute_single_fisher(embs[i], preds[i])
        return total_fisher

    def compute_single_fisher(self, x, p, ret_embed=False):
        p = p / np.sum(p)

        prob_matrix = np.diag(p) - np.outer(p, p)
        prob_matrix = self.orthogonal_transform @ prob_matrix @ self.orthogonal_transform
        prob_matrix = prob_matrix[1:,1:]

        use_regression_not_classification = False

        if use_regression_not_classification:
            if ret_embed:
                return np.expand_dims(x,1)
            else:
                return np.outer(x, x)
        else:
            if ret_embed:
                sqrt_prob_matrix = mysqrtm(prob_matrix) 

                return np.kron(sqrt_prob_matrix, np.expand_dims(x,1))
            else:
                return np.kron(prob_matrix, np.outer(x, x))




