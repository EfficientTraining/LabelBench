"""
Adapted from https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/algorithms/softmatch.
"""
import torch
import torch.nn.functional as F
import numpy as np
from LabelBench.trainer.trainer_impl.pytorch_semi_trainer import PyTorchSemiTrainer


class DistAlignEMAHook:
    """
    Distribution Alignment Hook for conducting distribution alignment
    """

    def __init__(self, num_classes, momentum=0.999, p_target_type='uniform', p_target=None):
        super().__init__()
        self.num_classes = num_classes
        self.m = momentum

        # p_target
        self.update_p_target, self.p_target = self.set_p_target(p_target_type, p_target)
        # p_model
        self.p_model = None

    @torch.no_grad()
    def dist_align(self, probs_x_ulb, probs_x_lb=None):
        # update queue
        self.update_p(probs_x_ulb, probs_x_lb)

        # dist align
        probs_x_ulb_aligned = probs_x_ulb * (self.p_target + 1e-6) / (self.p_model + 1e-6)
        probs_x_ulb_aligned = probs_x_ulb_aligned / probs_x_ulb_aligned.sum(dim=-1, keepdim=True)
        return probs_x_ulb_aligned

    @torch.no_grad()
    def update_p(self, probs_x_ulb, probs_x_lb):
        # check device
        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(probs_x_ulb.device)

        probs_x_ulb = probs_x_ulb.detach()
        if self.p_model == None:
            self.p_model = torch.mean(probs_x_ulb, dim=0)
        else:
            self.p_model = self.p_model * self.m + torch.mean(probs_x_ulb, dim=0) * (1 - self.m)

        if self.update_p_target:
            assert probs_x_lb is not None
            self.p_target = self.p_target * self.m + torch.mean(probs_x_lb, dim=0) * (1 - self.m)

    def set_p_target(self, p_target_type='uniform', p_target=None):
        assert p_target_type in ['uniform', 'gt', 'model']

        # p_target
        update_p_target = False
        if p_target_type == 'uniform':
            p_target = torch.ones((self.num_classes,)) / self.num_classes
        elif p_target_type == 'model':
            p_target = torch.ones((self.num_classes,)) / self.num_classes
            update_p_target = True
        else:
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)

        return update_p_target, p_target


class SoftMatchWeightingHook:
    """
    SoftMatch learnable truncated Gaussian weighting
    """

    def __init__(self, num_classes, n_sigma=2, momentum=0.999, per_class=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.n_sigma = n_sigma
        self.per_class = per_class
        self.m = momentum

        # initialize Gaussian mean and variance
        if not self.per_class:
            self.prob_max_mu_t = torch.tensor(1.0 / self.num_classes)
            self.prob_max_var_t = torch.tensor(1.0)
        else:
            self.prob_max_mu_t = torch.ones(self.num_classes) / self.args.num_classes
            self.prob_max_var_t = torch.ones(self.num_classes)

    @torch.no_grad()
    def update(self, probs_x_ulb):
        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        if not self.per_class:
            prob_max_mu_t = torch.mean(max_probs)  # torch.quantile(max_probs, 0.5)
            prob_max_var_t = torch.var(max_probs, unbiased=True)
            self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t.item()
            self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t.item()
        else:
            prob_max_mu_t = torch.zeros_like(self.prob_max_mu_t)
            prob_max_var_t = torch.ones_like(self.prob_max_var_t)
            for i in range(self.num_classes):
                prob = max_probs[max_idx == i]
                if len(prob) > 1:
                    prob_max_mu_t[i] = torch.mean(prob)
                    prob_max_var_t[i] = torch.var(prob, unbiased=True)
            self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t
            self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t
        return max_probs, max_idx

    @torch.no_grad()
    def masking(self, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
        if not self.prob_max_mu_t.is_cuda:
            self.prob_max_mu_t = self.prob_max_mu_t.to(logits_x_ulb.device)
        if not self.prob_max_var_t.is_cuda:
            self.prob_max_var_t = self.prob_max_var_t.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        self.update(probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        # compute weight
        if not self.per_class:
            mu = self.prob_max_mu_t
            var = self.prob_max_var_t
        else:
            mu = self.prob_max_mu_t[max_idx]
            var = self.prob_max_var_t[max_idx]
        mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (2 * var / (self.n_sigma ** 2))))
        return mask


class SoftmatchTrainer(PyTorchSemiTrainer):
    trainer_name = "softmatch"

    def __init__(self, trainer_config, dataset, model_fn, model_config, metric, get_feature_fn):
        """See `LabelBench.skeleton.trainer_skeleton.Trainer` for detailed documentation of the above arguments."""
        super().__init__(trainer_config, dataset, model_fn, model_config, metric, get_feature_fn)
        self.use_strong = True
        self.num_classes = self.dataset.get_num_classes()
        self.m = self.trainer_config["ema"]
        self.unlabeled_idxs = None
        self.ema_hook = None
        self.softmatch_hook = None

    def initialize_trainer(self):
        self.unlabeled_idxs = torch.from_numpy(self.dataset.unlabeled_idxs()).long()
        self.ema_hook = DistAlignEMAHook(self.num_classes, momentum=self.m,
                                         p_target_type='uniform' if self.trainer_config["dist_uniform"] else 'model')
        self.softmatch_hook = SoftMatchWeightingHook(self.num_classes, n_sigma=self.trainer_config["n_sigma"],
                                                     momentum=self.m)

    def train_step(self, model, img_l, target_l, class_weights, loss_fn, idx_u, img_uw, img_us, iter):
        # Get logits for labeled and unlabeled data.
        num_label = img_l.shape[0]
        num_unlabel = img_uw.shape[0]
        imgs = torch.cat([img_l, img_uw, img_us], dim=0).cuda()
        idx_u = idx_u.cuda()
        target_l = target_l.cuda()

        if not self.model_config["ret_emb"]:
            logits = model(imgs, ret_features=False).squeeze(-1)
        else:
            logits, _ = model(imgs, ret_features=True)
            logits = logits.squeeze(-1)
        logits_l = logits[:num_label]
        logits_uw = logits[num_label: (num_label + num_unlabel)]
        logits_us = logits[-num_unlabel:]

        # Construct supervised loss.
        if "weighted" in self.trainer_config and self.trainer_config["weighted"]:
            sup_loss = loss_fn(logits_l, target_l, weight=class_weights)
        else:
            sup_loss = loss_fn(logits_l, target_l)

        with torch.no_grad():
            probs_uw = F.softmax(logits_uw.data.float(), dim=-1)
            probs_l = F.softmax(logits_l.data.float(), dim=-1)
            probs_uw = self.ema_hook.dist_align(probs_x_ulb=probs_uw, probs_x_lb=probs_l)
            mask = self.softmatch_hook.masking(logits_x_ulb=probs_uw, softmax_x_ulb=False)
            max_probs, pseudo_label = torch.max(probs_uw, dim=-1)

        # Masked consistency loss (pseudo-labels of weakly augmented data applied to strongly augmented data).
        unsup_loss = F.cross_entropy(logits_us, pseudo_label.cuda(), reduction='none')
        unsup_loss = unsup_loss * mask.half().cuda()
        unsup_loss = unsup_loss.mean()

        return sup_loss + self.trainer_config["ulb_loss_ratio"] * unsup_loss
