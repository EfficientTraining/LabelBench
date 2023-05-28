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

    x, p = np.expand_dims(x, 1), np.expand_dims(p, 1) # x_single.size() = [embed_dim,1], p_single.size() = [n_classes,1]

    prob_matrix = np.diag(p) - np.outer(p,p)

    if False:
        if ret_embed:
            return x
        else:
            return np.outer(x,x)
    else:
        if ret_embed:
            return np.kron( scipy.linalg.sqrtm(prob_matrix), x)
        else:
            return np.kron( prob_matrix, np.outer(x,x) )

def woodbury(A_inv,U,sign):
    big_dim = A_inv.shape[0]
    small_dim = min(U.shape)

    if U.shape != (big_dim,small_dim):
        U = U.T

    outside = A_inv @ U
    inside = np.eye(small_dim) + sign * (U.T @ A_inv) @ U
    return A_inv - sign * outside @ np.linalg.inv(inside) @ outside.T


class BAIT(Strategy):
    strategy_name = "bait"

    def __init__(self, strategy_config, dataset):
        super(BAIT, self).__init__(strategy_config, dataset)
        self.input_types = {ALInput.TRAIN_PRED, ALInput.TRAIN_EMBEDDING}
        self.compute_single_hessian_fn = compute_single_hessian
        self.lamb = strategy_config["lamb"] if "lamb" in strategy_config else 1 # Regularization parameter for the fisher information matrix.
        self.if_fixed_step = strategy_config["fixed_step"] if "fixed_step" in strategy_config else False
        self.max_iter = strategy_config["max_iter"] if "max_iter" in strategy_config else 5 # Maximum number of iterations for Frank-Wolfe.
        self.fisher_comp_ratio = strategy_config["fisher_comp_ratio"] if "fisher_comp_ratio" in strategy_config else 1 # Instead of computing fisher for all samples, we only compute gradient for a random subset of samples to accelerate.
        self.grad_comp_ratio = strategy_config["grad_comp_ratio"] if "grad_comp_ratio" in strategy_config else 1 # Instead of computing gradient for all samples, we only compute gradient for a random subset of samples to accelerate.
        self.curInv = 0 # Current inverse of the fisher information matrix on the selected samples. 
        self.fisher_unlabeled = None # Fisher information matrix on the unlabeled set.
        self.batch_size = strategy_config["batch_size"] if "batch_size" in strategy_config else 1 # Batch size for computing the fisher information matrix.

        # The whole algorithm is trying to minimize the following optimization problem with fixed selection budget.
        # trace((self.curInv)@ self.fisher_unlabeled)

    def select(self, trainer, budget):
        print(budget)

        preds, embs = trainer.retrieve_inputs(self.input_types)
        probs = scipy.special.softmax(preds, axis=1)

        #PCA

        shifted_embs = embs - np.mean(embs, axis=0)
        cov = shifted_embs.T @ shifted_embs / shifted_embs.shape[0]
        v, w = np.linalg.eigh(cov)
        idx = (v.argsort()[::-1])[:50] # Sort descending and get top
        embs = embs @ w[:,idx]


        labeled_set = set(list(self.dataset.labeled_idxs()))
        all_set = set(list(range(len(self.dataset))))
        unlabeled = list(all_set - labeled_set)

        self.fisher_total = self.compute_fisher(range(len(self.dataset)) , probs, embs)
        self.fisher_total_sqrt = scipy.linalg.sqrtm( self.fisher_total )
        self.fisher_total_invsqrt = np.linalg.inv( self.fisher_total_sqrt )


        proposed_batch = list(np.random.choice(unlabeled, size=(budget,), replace=False))

        self.inv_fisher_proposed_adj  =  self.fisher_total_sqrt @ np.linalg.inv( self.compute_fisher(proposed_batch, probs, embs) ) @ self.fisher_total_sqrt
        self.cur_val = np.trace( self.inv_fisher_proposed_adj )
        print(self.cur_val)

        for _ in range(1):
            for _ in tqdm(range(budget)):
                swap_remove = proposed_batch[0]
                
                swap_adds = np.random.choice(list(set(unlabeled) - set(proposed_batch)), size=(10,), replace=False)


                remove_U = self.fisher_total_invsqrt @ self.compute_single_hessian_fn(embs[swap_remove], probs[swap_remove], ret_embed=True)
                inter_inv_fisher_adj = woodbury(self.inv_fisher_proposed_adj, remove_U, sign=-1)
               
                best_swap_add = swap_remove
                best_swap_add_val = self.cur_val
                best_swap_matrix = self.inv_fisher_proposed_adj
                for swap_add in swap_adds:
                    add_U = self.fisher_total_invsqrt @ self.compute_single_hessian_fn(embs[swap_add], probs[swap_add], ret_embed=True)

                    swap_inv_fisher_adj = woodbury(inter_inv_fisher_adj, add_U, sign=1)

                    swap_val = np.trace( swap_inv_fisher_adj )

                    if swap_val < best_swap_add_val:
                        best_swap_add = swap_add
                        best_swap_add_val = swap_val
                        best_swap_matrix = swap_inv_fisher_adj

                proposed_batch.append(best_swap_add)
                del proposed_batch[0]

                self.cur_val = best_swap_add_val
                self.inv_fisher_proposed_adj = best_swap_matrix

            

            print(self.cur_val)
            self.inv_fisher_proposed_adj  =  self.fisher_total_sqrt @ np.linalg.inv( self.compute_fisher(proposed_batch, probs, embs) ) @ self.fisher_total_sqrt
            self.cur_val = np.trace( self.inv_fisher_proposed_adj )
            print(self.cur_val)
        
        return proposed_batch



    def compute_fisher(self, indices, probs, embs, num_workers = 4):
        embs = embs[indices]
        probs = probs[indices]
        total_fisher = 0
        count = 0
        for i in tqdm(range(len(indices)), desc="Computing Fisher Information"):
            total_fisher = total_fisher + self.compute_single_hessian_fn(embs[i], probs[i])

        return total_fisher



