import torch
import torch.nn.functional as F
from ALBench.trainer.trainer_impl.pytorch_semi_trainer import PyTorchSemiTrainer


class FlexmatchTrainer(PyTorchSemiTrainer):
    trainer_name = "flexmatch"
    # adapted from https://github.com/microsoft/Semi-supervised-learning/tree/main/semilearn/algorithms/flexmatch

    def __init__(self, trainer_config, dataset, model_fn, model_config, metric, get_feature_fn):
        super().__init__(trainer_config, dataset, model_fn, model_config, metric, get_feature_fn)
        self.use_strong = True
        self.uhat = None
        self.sigma = None
        self.n_ul = None
        self.unlabeled_idxs = None

    def pretrain(self):
        self.unlabeled_idxs = torch.from_numpy(self.dataset.unlabeled_idxs()).long().cuda()
        self.uhat = -1 * \
            torch.ones(len(self.dataset), dtype=torch.long).cuda()
        self.sigma = torch.zeros(
            self.dataset.get_num_classes() + 1, dtype=torch.long).cuda()
        self.n_ul = len(self.unlabeled_idxs)
        self.sigma[-1] = self.n_ul

    def train_step(self, model, img_l, target_l, class_weights, loss_fn, idx_u, img_uw, img_us):
        # TODO: documentation
        # Get logits for labeled and unlabeled data.
        if not self.model_config["ret_emb"]:
            logits_l = model(img_l, ret_features=False).squeeze(-1)
            logits_uw = model(img_uw, ret_features=False).squeeze(-1)
            logits_us = model(img_us, ret_features=False).squeeze(-1)
        else:
            logits_l, _ = model(img_l, ret_features=True)
            logits_l = logits_l.squeeze(-1)
            logits_uw, _ = model(img_uw, ret_features=True)
            logits_uw = logits_uw.squeeze(-1)
            logits_us, _ = model(img_us, ret_features=True)
            logits_us = logits_us.squeeze(-1)

        # Construct supervised loss.
        if "weighted" in self.trainer_config and self.trainer_config["weighted"]:
            sup_loss = loss_fn(logits_l, target_l, weight=class_weights)
        else:
            sup_loss = loss_fn(logits_l, target_l)

        with torch.no_grad():

            # set mask for terms to appear in unsupervised loss, and select terms to update uhat and sigma
            beta = self.sigma[:-1] / torch.max(self.sigma)
            probs_uw = F.softmax(logits_uw, dim=-1)
            max_probs, pseudo_label = torch.max(probs_uw, dim=-1)
            mask = max_probs.ge(
                self.trainer_config['p_cutoff'] * (beta[pseudo_label] / (2. - beta[pseudo_label])))
            mask = mask.to(max_probs.dtype)
            select = max_probs.ge(self.trainer_config['p_cutoff'])

            # update uhat and sigma
            if idx_u[select == 1].nelement() != 0:
                self.uhat[idx_u[select == 1]] = pseudo_label[select == 1]

            for c in range(self.dataset.get_num_classes()):
                self.sigma[c] = torch.sum(torch.eq(self.uhat[self.unlabeled_idxs], c))
            self.sigma[-1] = self.n_ul - torch.sum(self.sigma[:-1])

        # Masked consistency loss (pseudo-labels of weakly augmented data applied to strongly augmented data).
        unsup_loss = F.cross_entropy(logits_us, pseudo_label, reduction='none')
        unsup_loss = unsup_loss * mask
        unsup_loss = unsup_loss.mean()

        return sup_loss + self.trainer_config["ulb_loss_ratio"] * unsup_loss
