
from copy import deepcopy
import gc
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from functorch import vmap
from torch.utils.data import DataLoader, Subset
from ALBench.skeleton.active_learning_skeleton import Strategy, ALInput

# Compute the covariance of gradient for one sample as the approximate hessian.
def compute_single_hessian(x,p,weight=None, ret_embed=False):
    
    x, p = x.unsqueeze(1), p.unsqueeze(1) # x_single.size() = [embed_dim,1], p_single.size() = [n_classes,1]
    fisher_single = torch.kron(torch.diag(p)-p@p.mT,x@x.mT)
    fisher_single = fisher_single*weight
    return fisher_single

# Approximate the fisher information for one sample from gradient
def compute_single_hessian_withExpGradEmbedding(x, p, weight=None, ret_embed=False):   
    x, p = x.unsqueeze(1), p.unsqueeze(1) # x_single.size() = [embed_dim,1], p_single.size() = [n_classes,1]
    # Compute a n_class number of gradient for cross-entropy loss. 
    # The explicit form of grad of [L(x,y, W)]_{w_i} = \one[i = y]x + p_i * x. Each column corresponds to a class (y).
    exp_grad_embed = torch.kron(torch.eye(p.shape[0]).cuda() - p.repeat(1,p.shape[0]),x)
    # Scale the gradient (each column) with the square root of the probability of each class. 
    # Therefore, exp_grad_embed @ exp_grad_embed.mT leads to the expectation over y.
    # The explicit form of each column: [\sqrt{p_y}\one[i = y]x + \sqrt{p_y} p_i * x]_{i=1,...,n_class}. 
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
        self.input_types = {ALInput.TRAIN_PRED}
        self.unlabeled = set(list(np.arange(len(dataset)))) 
        assert strategy_config["use_embeddings"], "Current version of BAIT only supports embedding dataset."
        self.dataset = dataset
        self.compute_single_hessian_fn = compute_single_hessian_withExpGradEmbedding
        self.lamb = strategy_config["lamb"] if "lamb" in strategy_config else 1
        self.ifLinearSearch = strategy_config["linear_search"] if "linear_search" in strategy_config else False
        self.max_iter = strategy_config["max_iter"] if "max_iter" in strategy_config else 5 # Maximum number of iterations for Frank-Wolfe.
        self.fisher_comp_ratio = strategy_config["fisher_comp_ratio"] if "fisher_comp_ratio" in strategy_config else 1 # Instead of computing fisher for all samples, we only compute gradient for a random subset of samples.
        self.grad_comp_ratio = strategy_config["grad_comp_ratio"] if "grad_comp_ratio" in strategy_config else 1 # Instead of computing gradient for all samples, we only compute gradient for a random subset of samples.
        self.curInv = 0 # Current inverse of the fisher information matrix on the selected samples.
        self.fisher_unlabeled = None # Fisher information matrix on the unlabeled set.
        self.train_dataset = None # Training dataset for computing fisher information.

    def select(self, trainer, budget):

        self.unlabeled = self.unlabeled - set(list(self.dataset.labeled_idxs()))
        self.train_dataset,_,_ = self.dataset.get_embedding_datasets()
        all_set = set(list(range(len(self.dataset))))
        all_preds = torch.Tensor(trainer.retrieve_inputs(self.input_types)[0]).cuda()
 
        ## Update the fisher information for unlabeled set.
        # Instead of computing grad for all the unlabeled samples,
        # we randomly sample a portion of the samples to compute grad and optimize.
        fisher_est_indices = np.random.choice(list(self.unlabeled), len(self.unlabeled)//self.fisher_comp_ratio, replace=False) #TODO: 3
        self.fisher_unlabeled = self.compute_fisher(fisher_est_indices, all_preds)
        ## Compute the distribution for unlabeled set by running Frank-Wolfe.
        select_distribution = self.frank_wolfe(all_preds, budget)

        # ## Verify if the inverse computed by addition method is same as the true inverse. Not used in real algorithm.
        # curFisher = self.compute_fisher(list(self.unlabeled), all_preds, weights=select_distribution[np.array(list(self.unlabeled))]) \
        #     + torch.eye(self.fisher_unlabeled.shape[0]).cuda()*self.lamb
        # currentValue = torch.trace(torch.linalg.solve(curFisher, self.fisher_unlabeled))
        # print("Final Current Value: {}".format(currentValue))

        selected = np.random.choice(list(all_set), budget, p=select_distribution/select_distribution.sum(), replace=False)
        return selected
    
    def frank_wolfe(self, all_preds, budget, tol = 0):

        # Initialize the first indices set by random sampling.
        next_indices = np.random.choice(list(self.unlabeled), budget, replace=False)
        step = 0        
        
        # Start the iteration.
        while step <= 2 or (- topK_vals[0] > tol and step <= self.max_iter): #TODO 
            #### Run the linear search to find the proper step size alpha and the corresponding inverse fisher.
            alpha = self.linearSearch(base_beta=2/(2+step), new_indices=next_indices, all_preds=all_preds, ifLinearSearch=self.ifLinearSearch if step > 5 else True)
            next_distribution = (1-alpha)*next_distribution if step > 0 else np.zeros(len(self.dataset)) #distribution over all dataset
            next_distribution[next_indices] += alpha

            #### Find the largest gradient coordinates and query them in the next iteration.
            # Instead of computing grad for all the unlabeled samples,
            # we randomly sample a portion of the samples to compute grad and optimize.
            grad_est_indices = np.random.choice(list(self.unlabeled), len(self.unlabeled)//self.grad_comp_ratio, replace=False) #TODO: 3
            # Compute the grads for the  grad_est_indices instead of all unlabeled indices
            grads = self.compute_grad_forOP(grad_est_indices, all_preds)
            # Choose the k-minimum gradient. Note grads are negative so k-minimum equals k largest magnitude.
            topK_vals, topK_indices = torch.topk(torch.Tensor(grads), budget, largest=False)
            next_indices = grad_est_indices[np.array(topK_indices.tolist())]

            step+=1
            print("Step: {}, alpha: {}, max |grad|: {}".format(step, alpha, topK_vals[0]))
            del grads
            torch.cuda.empty_cache()

        del self.curInv
        torch.cuda.empty_cache()

        return  next_distribution
    
    def compute_fisher(self, indices, all_preds, weights = None, batch_size = 1, num_workers = 4):
        if weights is None:  
            weights = np.ones_like(indices)
        else:
            assert len(weights) == len(indices), "Length of weights should be the same as length of indices."
        weights = torch.Tensor(weights).cuda()
        
        dataset = Subset(self.train_dataset, indices)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        preds = all_preds[indices]
        total_fisher = 0
        count = 0
        print("Computing Fisher Information for given data ...")
        for embed_feature, _, *other in tqdm(loader, desc="Batch Index"):
            embed_feature = embed_feature.float().cuda()
            pred = preds[count:count+len(embed_feature)]
            prob = F.softmax(pred, dim=1)
            fisher_batch = vmap(self.compute_single_hessian_fn)(embed_feature, prob, weights[count:count+len(embed_feature)])
            total_fisher = total_fisher + torch.sum(fisher_batch, dim=0)
            count += len(embed_feature)

        total_fisher = total_fisher / weights.sum()
        del fisher_batch
        torch.cuda.empty_cache()
        gc.collect()
        
        return total_fisher
    
    
    def update_inverse(self, alpha, new_indices, all_preds, batch_size = 1, num_workers = 4):
        dataset = Subset(self.train_dataset, new_indices)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        preds = all_preds[new_indices]
        count = 0
        print("Updating the inverse for given data ...")
        self.curInv = (1/(1-alpha))*self.curInv

        # Add the new fisher into inverse by using woodbury formula one by one.
        for embed_feature, _, *other in tqdm(loader, desc="Batch Index"):
            embed_feature = embed_feature.float().cuda()
            pred = preds[count:count+len(embed_feature)]
            prob = F.softmax(pred, dim=1)
            exp_grad_embed = vmap(self.compute_single_hessian_fn)(embed_feature, prob, ret_embed=True)
            exp_grad_embed = exp_grad_embed.view(exp_grad_embed.shape[0]*exp_grad_embed.shape[1], exp_grad_embed.shape[2])
            self.curInv = self.curInv - alpha/len(new_indices)*self.curInv@exp_grad_embed.mT\
                @torch.linalg.solve(torch.eye(exp_grad_embed.shape[0]).cuda() +  alpha/len(new_indices)*exp_grad_embed@self.curInv@exp_grad_embed.mT, exp_grad_embed@self.curInv) 
            torch.cuda.empty_cache()
            count += len(embed_feature)

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

    
    def compute_grad_forOP(self, indices, all_preds, batch_size = 1, num_workers = 4):
        dataset = Subset(self.train_dataset, indices)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        preds = all_preds[indices]
        count = 0
        grads = []
        print("Computing Gradient for the target function inv(V)U on each given sample ...")
        for embed_feature, _, *other in tqdm(loader, desc="Batch Index"):
            embed_feature = embed_feature.float().cuda()
            pred = preds[count:count+len(embed_feature)]
            prob = F.softmax(pred, dim=1)
            # Compute the gradient for one sample.
            def compute_single_grad(x,p):
                exp_grad_embed_single = self.compute_single_hessian_fn(x,p,1,ret_embed=True).float()
                grad = - torch.trace(exp_grad_embed_single@self.curInv@self.fisher_unlabeled@self.curInv@exp_grad_embed_single.mT)
                return grad
            batched_grad = vmap(compute_single_grad)(embed_feature,prob)
            grads.extend(batched_grad.tolist())
            count += len(embed_feature)
        return grads
    
    def linearSearch(self, base_beta, new_indices, all_preds, ifLinearSearch):
        alpha = base_beta
        
        # Randomly sample a subset and initialize its regularized inverse fisher.
        if base_beta == 1:
            newFisher_regu = self.compute_fisher(new_indices, all_preds) + torch.eye(self.fisher_unlabeled.shape[0]).cuda()*self.lamb
            print("Updating the inverse for given data ...")
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
                self.update_inverse(inner_alpha, new_indices, all_preds)
                currentValue = torch.trace(self.curInv@self.fisher_unlabeled)
                print("With alpha = {}, Current Value: {}".format(alpha, currentValue))
                alpha +=  base_beta
                torch.cuda.empty_cache()
                # If oneStep, then we use pre-scheduled step size (e.g. 2/(2+step)).
                # Otherwise, continue while loop to do linear search to find the best alpha.
                if ifLinearSearch: break
            alpha -= base_beta
        return alpha
    

    

    
    