import torch
import torch.nn.functional as F
from LabelBench.trainer.trainer_impl.pytorch_semi_trainer import PyTorchSemiTrainer
import numpy as np


class PseudolabelTrainer(PyTorchSemiTrainer):
    trainer_name = "pseudolabel"
    # Adapted from https://github.com/microsoft/Semi-supervised-learning/tree/main/semilearn/algorithms/pseudolabel.

    def __init__(self, trainer_config, dataset, model_fn, model_config, metric, get_feature_fn):
        """See `LabelBench.skeleton.trainer_skeleton.Trainer` for detailed documentation of the above arguments."""
        super().__init__(trainer_config, dataset, model_fn, model_config, metric, get_feature_fn)
        self.use_strong = False
        self.unlabeled_idxs = None

    def initialize_trainer(self):
        self.unlabeled_idxs = torch.from_numpy(self.dataset.unlabeled_idxs()).long()

    def train_step(self, model, img_l, target_l, class_weights, loss_fn, idx_u, img_uw, img_us, iter):
        # Get logits for labeled and unlabeled data.
        num_label = img_l.shape[0]
        num_unlabel = img_uw.shape[0]
        imgs = torch.cat([img_l, img_uw], dim=0).cuda()
        target_l = target_l.cuda()

        if not self.model_config["ret_emb"]:
            logits = model(imgs, ret_features=False).squeeze(-1)
        else:
            logits, _ = model(imgs, ret_features=True)
            logits = logits.squeeze(-1)
        logits_l = logits[:num_label]
        logits_uw = logits[-num_label:]

        # Construct supervised loss.
        if "weighted" in self.trainer_config and self.trainer_config["weighted"]:
            sup_loss = loss_fn(logits_l, target_l, weight=class_weights)
        else:
            sup_loss = loss_fn(logits_l, target_l)

        with torch.no_grad():
            # Select high confidence samples
            probs_uw = F.softmax(logits_uw.data.cpu().float(), dim=-1)
            max_probs, pseudo_label = torch.max(probs_uw, dim=-1)
            select = max_probs.ge(self.trainer_config['p_cutoff'])
            pseudo_label, logits_uw = pseudo_label[select], logits_uw[select]

        # Compute unsupervised loss
        unsup_warmup = np.clip(iter/self.trainer_config['ulb_warm_up'],  a_min=0.0, a_max=1.0)
        unsup_loss = F.cross_entropy(logits_uw, pseudo_label.cuda(), reduction='mean')

        return sup_loss + self.trainer_config["ulb_loss_ratio"] * unsup_warmup * unsup_loss
