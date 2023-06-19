import torch
import torch.nn.functional as F
from LabelBench.trainer.trainer_impl.pytorch_semi_trainer import PyTorchSemiTrainer


class FlexmatchTrainer(PyTorchSemiTrainer):
    trainer_name = "flexmatch"
    # Adapted from https://github.com/microsoft/Semi-supervised-learning/tree/main/semilearn/algorithms/flexmatch.

    def __init__(self, trainer_config, dataset, model_fn, model_config, metric, get_feature_fn):
        """See `LabelBench.skeleton.trainer_skeleton.Trainer` for detailed documentation of the above arguments."""
        super().__init__(trainer_config, dataset, model_fn, model_config, metric, get_feature_fn)
        self.use_strong = True
        self.uhat = None
        self.sigma = None
        self.n_ul = None
        self.unlabeled_idxs = None

    def initialize_trainer(self):
        self.unlabeled_idxs = torch.from_numpy(self.dataset.unlabeled_idxs()).long()
        self.uhat = -1 * torch.ones(len(self.dataset), dtype=torch.long)
        self.sigma = torch.zeros(self.dataset.get_num_classes() + 1, dtype=torch.long)
        self.n_ul = len(self.unlabeled_idxs)
        self.sigma[-1] = self.n_ul

    def train_step(self, model, img_l, target_l, class_weights, loss_fn, idx_u, img_uw, img_us):
        # TODO: documentation
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
            # Set mask for terms to appear in unsupervised loss, and select terms to update uhat and sigma.
            beta = self.sigma[:-1] / torch.max(self.sigma)
            probs_uw = F.softmax(logits_uw.data.cpu().float(), dim=-1)
            max_probs, pseudo_label = torch.max(probs_uw, dim=-1)
            mask = max_probs.ge(self.trainer_config['p_cutoff'] * (beta[pseudo_label] / (2. - beta[pseudo_label])))
            mask = mask.to(max_probs.dtype)
            select = max_probs.ge(self.trainer_config['p_cutoff'])

            # Update uhat and sigma.
            if idx_u[select].nelement() != 0:
                self.uhat[idx_u[select]] = pseudo_label[select]

            for c in range(self.dataset.get_num_classes()):
                self.sigma[c] = torch.sum(torch.eq(self.uhat[self.unlabeled_idxs], c))
            self.sigma[-1] = self.n_ul - torch.sum(self.sigma[:-1])

        # Masked consistency loss (pseudo-labels of weakly augmented data applied to strongly augmented data).
        unsup_loss = F.cross_entropy(logits_us, pseudo_label.cuda(), reduction='none')
        unsup_loss = unsup_loss * mask.half().cuda()
        unsup_loss = unsup_loss.mean()

        return sup_loss + self.trainer_config["ulb_loss_ratio"] * unsup_loss
