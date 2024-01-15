import torch
import torch.nn.functional as F
from LabelBench.trainer.trainer_impl.pytorch_semi_trainer import PyTorchSemiTrainer


def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val


def entropy_loss(mask, logits_s, prob_model, label_hist):
    mask = mask.bool()

    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()


class FreematchTrainer(PyTorchSemiTrainer):
    trainer_name = "freematch"
    # Adapted from https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/algorithms/freematch.

    def __init__(self, trainer_config, dataset, model_fn, model_config, metric, get_feature_fn):
        """See `LabelBench.skeleton.trainer_skeleton.Trainer` for detailed documentation of the above arguments."""
        super().__init__(trainer_config, dataset, model_fn, model_config, metric, get_feature_fn)
        self.use_strong = True
        self.num_classes = self.dataset.get_num_classes()
        self.m = self.trainer_config["ema_p"]
        self.unlabeled_idxs = None
        self.p_model = None
        self.label_hist = None
        self.time_p = None

    def initialize_trainer(self):
        self.unlabeled_idxs = torch.from_numpy(self.dataset.unlabeled_idxs()).long()
        self.p_model = torch.ones(self.num_classes) / self.num_classes
        self.label_hist = torch.ones(self.num_classes) / self.num_classes
        self.time_p = self.p_model.mean()

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
            if not self.p_model.is_cuda:
                self.p_model = self.p_model.to(logits_uw.device)
            if not self.label_hist.is_cuda:
                self.label_hist = self.label_hist.to(logits_uw.device)
            if not self.time_p.is_cuda:
                self.time_p = self.time_p.to(logits_uw.device)

            probs_uw = F.softmax(logits_uw.data.float(), dim=-1)
            max_probs, max_idx = torch.max(probs_uw, dim=-1, keepdim=True)

            self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean()

            if self.trainer_config["clip_thresh"]:
                self.time_p = torch.clip(self.time_p, 0.0, 0.95)

            self.p_model = self.p_model * self.m + (1 - self.m) * probs_uw.mean(dim=0)
            hist = torch.bincount(max_idx.reshape(-1), minlength=self.p_model.shape[0]).to(self.p_model.dtype)
            self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())

            max_probs, pseudo_label = torch.max(probs_uw, dim=-1)
            mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
            mask = max_probs.ge(self.time_p * mod[pseudo_label]).to(max_probs.dtype)

        # Masked consistency loss (pseudo-labels of weakly augmented data applied to strongly augmented data).
        unsup_loss = F.cross_entropy(logits_us, pseudo_label.cuda(), reduction='none')
        unsup_loss = unsup_loss * mask.half().cuda()
        unsup_loss = unsup_loss.mean()

        # Calculate entropy loss.
        if mask.sum() > 0:
            ent_loss, _ = entropy_loss(mask.half(), logits_us, self.p_model.half(), self.label_hist.half())
        else:
            ent_loss = 0.0

        return sup_loss + self.trainer_config["ulb_loss_ratio"] * unsup_loss + \
            self.trainer_config["ent_loss_ratio"] * ent_loss
