from copy import deepcopy
import gc
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from functorch import vmap
from ALBench.skeleton.active_learning_skeleton import Strategy, ALInput


# Approximate the fisher information for one sample from gradient by using the covariance of gradient.
def compute_single_hessian_withExpGradEmbedding(x, p, weight=None, ret_embed=False):   

    # Currently, weight is only used for sanity check when computing the fisher information and make sure it consistent with the results from woodbury formula.

    x, p = x.unsqueeze(1), p.unsqueeze(1) # x_single.size() = [embed_dim,1], p_single.size() = [n_classes,1]
    # Compute a n_class number of gradient for cross-entropy loss. 
    # The explicit form of grad of [L(x,y, W)]_{w_i} = \one[i = y]x + p_i * x. Each column corresponds to a class (y).
    exp_grad_embed = torch.kron(torch.eye(p.shape[0]).cuda() - p.repeat(1,p.shape[0]), x)
    # Scale the gradient (each column) with the square root of the probability of each class. 
    # Therefore, exp_grad_embed @ exp_grad_embed.mT leads to the expectation over y.
    # The explicit form of each column : [\sqrt{p_y}\one[i = y]x + \sqrt{p_y} p_i * x]_{i=1,...,n_class}. 
    exp_grad_embed = exp_grad_embed * torch.sqrt(p.expand(p.shape[0],exp_grad_embed.shape[0]).mT)
    if ret_embed:
        # Return the low-rank expected gradient embedding matrix with size [n_classes,n_classes*embed_dim, ]
        return exp_grad_embed.mT
    else:
        # Return the fisher information matrix with size [n_classes*embed_dim,n_classes*embed_dim]
        fisher_single = exp_grad_embed @ exp_grad_embed.mT
        fisher_single = fisher_single*weight
        return fisher_single


class BAIT(Strategy):
    strategy_name = "bait"

    def __init__(self, strategy_config, dataset):
        super(BAIT, self).__init__(strategy_config, dataset)
        self.input_types = {ALInput.TRAIN_PRED, ALInput.TRAIN_EMBEDDING}
        self.compute_single_hessian_fn = compute_single_hessian_withExpGradEmbedding
        self.lamb = strategy_config["lamb"] if "lamb" in strategy_config else 1 # Regularization parameter for the fisher information matrix.
        self.if_fixed_step = strategy_config["fixed_step"] if "fixed_step" in strategy_config else False
        self.max_iter = strategy_config["max_iter"] if "max_iter" in strategy_config else 5 # Maximum number of iterations for Frank-Wolfe.
        self.fisher_comp_ratio = strategy_config["fisher_comp_ratio"] if "fisher_comp_ratio" in strategy_config else 1 # Instead of computing fisher for all samples, we only compute gradient for a random subset of samples to accelerate.
        self.grad_comp_ratio = strategy_config["grad_comp_ratio"] if "grad_comp_ratio" in strategy_config else 1 # Instead of computing gradient for all samples, we only compute gradient for a random subset of samples to accelerate.
        self.curInv = 0 # Current inverse of the fisher information matrix on the selected samples. 
        self.fisher_unlabeled = None # Fisher information matrix on the unlabeled set.

        # The whole algorithm is trying to minimize the following optimization problem with fixed selection budget.
        # trace((self.curInv)@ self.fisher_unlabeled)

    def select(self, trainer, budget):
        preds, embs = trainer.retrieve_inputs(self.input_types)
        labeled_set = set(list(self.dataset.labeled_idxs()))
        all_set = set(list(range(len(self.dataset))))
        unlabeled = np.array(list(all_set - labeled_set))
        unlabeled_set = set(unlabeled)

        ## Update the fisher information for unlabeled set.
        # Instead of computing grad for all the unlabeled samples,
        # we randomly sample a portion of the samples to compute grad and optimize.
        fisher_est_indices = np.random.choice(unlabeled, len(unlabeled)//self.fisher_comp_ratio, replace=False) #TODO: 3
        self.fisher_unlabeled = self.compute_fisher(fisher_est_indices, preds, embs)
        ## Compute the distribution for unlabeled set by running Frank-Wolfe.
        select_distribution = self.frank_wolfe(preds, embs, unlabeled, budget)

        # ## Sanity check: Verify if the inverse computed by addition method is same as the true inverse. Not used in real algorithm.
        # curFisher = self.compute_fisher(unlabeled, preds, embs, weights=select_distribution[unlabeled]) \
        #     + torch.eye(self.fisher_unlabeled.shape[0]).cuda()*self.lamb
        # currentValue = torch.trace(torch.linalg.solve(curFisher, self.fisher_unlabeled))
        # print("Final Current Value: {}".format(currentValue))

        selected = np.random.choice(list(all_set), budget, p=select_distribution/select_distribution.sum(), replace=False)
        return selected
    
    def compute_fisher(self, indices, preds, embs, weights = None, batch_size = 1, num_workers = 4):
        if weights is None:  
            weights = np.ones_like(indices)
        else:
            assert len(weights) == len(indices), "Length of weights should be the same as length of indices."
        weights = torch.Tensor(weights).cuda()
        
        embs = torch.from_numpy(embs[indices])
        preds = torch.from_numpy(preds[indices])
        total_fisher = 0
        count = 0
        # print("Computing Fisher Information for given data ...")
        for i in tqdm(range(0, len(indices), batch_size)):
            batch_embs = embs[i: min(i+batch_size,len(indices))].cuda().float()
            batch_preds = preds[i: min(i+batch_size,len(indices))].cuda().float()
            batch_probs = F.softmax(batch_preds, dim=1)
            batch_fisher = vmap(self.compute_single_hessian_fn)(batch_embs, batch_probs, weights[count:count+len(batch_embs)])
            total_fisher = total_fisher + torch.sum(batch_fisher, dim=0)
            count += len(batch_embs)

        total_fisher = total_fisher / weights.sum()
        del batch_fisher
        torch.cuda.empty_cache()
        gc.collect()
        
        return total_fisher
    
    def frank_wolfe(self, preds, embs,  unlabeled, budget, tol = 0):

        # Initialize the first indices set by random sampling.
        next_indices = np.random.choice(unlabeled, budget, replace=False)
        step = 0        
        
        # Start the iteration.
        while step <= 2 or (- topK_vals[0] > tol and step <= self.max_iter): #TODO 
            #### Run the linear search to find the proper step size alpha and the corresponding inverse fisher.
            # Another option is to use other search methods, e.g., binary search. But we find linear search might be enough.
            alpha = self.linearSearch(2/(2+step), next_indices, preds, embs, if_fixed_step=self.if_fixed_step if step > 5 else True)
            next_distribution = (1-alpha)*next_distribution if step > 0 else np.zeros(len(self.dataset)) #distribution over all dataset
            next_distribution[next_indices] += alpha

            #### Find the largest gradient coordinates and query them in the next iteration.
            # Instead of computing grad for all the unlabeled samples,
            # we randomly sample a portion of the samples to compute grad and optimize.
            grad_est_indices = np.random.choice(unlabeled, len(unlabeled)//self.grad_comp_ratio, replace=False) #TODO: 3
            # Compute the grads for the  grad_est_indices instead of all unlabeled indices
            grads = self.compute_grad_forOP(grad_est_indices, preds, embs)
            # Choose the k-minimum gradient. Note grads are negative so k-minimum equals k largest magnitude.
            topK_vals, topK_indices = torch.topk(torch.Tensor(grads), budget, largest=False)
            next_indices = grad_est_indices[np.array(topK_indices.tolist())]

            step+=1
            # print("Step: {}, alpha: {}, max |grad|: {}".format(step, alpha, topK_vals[0]))
            del grads
            torch.cuda.empty_cache()

        del self.curInv
        torch.cuda.empty_cache()

        return  next_distribution
    
    def linearSearch(self, base_beta, new_indices, preds, embs, if_fixed_step):
        alpha = base_beta
        
        # Randomly sample a subset and initialize its regularized inverse fisher.
        if base_beta == 1:
            newFisher_regu = self.compute_fisher(new_indices, preds, embs) + torch.eye(self.fisher_unlabeled.shape[0]).cuda()*self.lamb
            # print("Updating the inverse for given data ...")
            self.curInv = torch.linalg.inv(newFisher_regu)
            currentValue = torch.trace(self.curInv@self.fisher_unlabeled)
            print("Current Value: {}".format(currentValue))
            del newFisher_regu
            torch.cuda.empty_cache()
        else:
            prevValue = None
            currentValue = None
            while alpha < 1 and (prevValue is None or currentValue < prevValue):
                inner_alpha = base_beta/(1 - (alpha - base_beta))
                prevValue = currentValue
                self.update_inverse(inner_alpha, new_indices, preds, embs)
                currentValue = torch.trace(self.curInv@self.fisher_unlabeled)
                print("With alpha = {}, Current Value: {}".format(alpha, currentValue))
                alpha +=  base_beta
                torch.cuda.empty_cache()
                # If oneStep, then we use pre-scheduled step size (e.g. 2/(2+step)).
                # Otherwise, continue while loop to do linear search to find the best alpha.
                if if_fixed_step: break
            alpha -= base_beta
        return alpha
    
    def update_inverse(self, alpha, new_indices, preds, embs, batch_size = 1, num_workers = 4):

        embs = torch.from_numpy(embs[new_indices])
        preds = torch.from_numpy(preds[new_indices])
        count = 0
        print("Updating the inverse for given data ...")
        self.curInv = (1/(1-alpha))*self.curInv

        # Add the new fisher into inverse by using woodbury formula one by one.
        for i in tqdm(range(0, len(new_indices), batch_size)):
            batch_embs = embs[i: min(i+batch_size,len(new_indices))].cuda().float()
            batch_preds = preds[i: min(i+batch_size,len(new_indices))].cuda().float()
            batch_probs = F.softmax(batch_preds, dim=1)
            exp_grad_embed = vmap(self.compute_single_hessian_fn)(batch_embs, batch_probs, ret_embed=True)
            exp_grad_embed = exp_grad_embed.view(exp_grad_embed.shape[0]*exp_grad_embed.shape[1], exp_grad_embed.shape[2])
            self.curInv = self.curInv - alpha/len(new_indices)*self.curInv@exp_grad_embed.mT\
                @torch.linalg.solve(torch.eye(exp_grad_embed.shape[0]).cuda() +  alpha/len(new_indices)*exp_grad_embed@self.curInv@exp_grad_embed.mT, exp_grad_embed@self.curInv) 
            torch.cuda.empty_cache()
            count += len(batch_embs)

        # Because we care about the (fisher + \lambda*I)^{-1} and we have applied ((1-\alpha)*(fisher + \lambda*I))^{-1},
        # we need to add \alpha*\lambda*I back to the inverse.
        # We use the same approach as above to add the one-hot vectors one by one.
        for i in tqdm(range(exp_grad_embed.shape[1]//exp_grad_embed.shape[0]), desc="Reg Batch Index"):
            tmp = torch.zeros(exp_grad_embed.shape[1],exp_grad_embed.shape[0]).cuda()
            tmp[i*exp_grad_embed.shape[0]:(i+1)*exp_grad_embed.shape[0],:] = torch.eye(exp_grad_embed.shape[0]).cuda()
            self.curInv = self.curInv - alpha*self.lamb*self.curInv@tmp\
                        @torch.linalg.solve(torch.eye(exp_grad_embed.shape[0]).cuda()+ alpha*self.lamb*tmp.mT@self.curInv@tmp, tmp.mT@self.curInv) 
            del tmp
            torch.cuda.empty_cache()

        del exp_grad_embed
        torch.cuda.empty_cache()
        gc.collect()

    
    def compute_grad_forOP(self, indices, preds, embs, batch_size = 1, num_workers = 4):

        embs = torch.from_numpy(embs[indices])
        preds = torch.from_numpy(preds[indices])
        count = 0
        grads = []
        print("Computing Gradient for the target function inv(V)U on each given sample ...")

        # Add the new fisher into inverse by using woodbury formula one by one.
        for i in tqdm(range(0, len(indices), batch_size)):
            batch_embs = embs[i: min(i+batch_size,len(indices))].cuda().float()
            batch_preds = preds[i: min(i+batch_size,len(indices))].cuda().float()
            batch_probs = F.softmax(batch_preds, dim=1)
            def compute_single_grad(x, p):
                exp_grad_embed_single = self.compute_single_hessian_fn(x, p, 1,ret_embed=True).float()
                grad = - torch.trace(exp_grad_embed_single@self.curInv@self.fisher_unlabeled@self.curInv@exp_grad_embed_single.mT)
                return grad
            batched_grad = vmap(compute_single_grad)(batch_embs, batch_probs)
            grads.extend(batched_grad.tolist())
            count += len(batch_embs)
        return grads


    

    
    