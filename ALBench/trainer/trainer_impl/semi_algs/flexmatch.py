import torch

from ..pytorch_semi_trainer import PyTorchSemiTrainer # TODO: is this right?

class FlexmatchTrainer(PyTorchSemiTrainer):
    trainer_name = "flexmatch" # TODO: is this being set correctly, will the parent trainer_name be overridden?
    # directly modified from https://github.com/microsoft/Semi-supervised-learning/tree/main/semilearn/algorithms/flexmatch

    def __init__(self, trainer_config, dataset, model_fn, model_config, metric, get_feature_fn):
        super().__init__(trainer_config, dataset, model_fn, model_config, metric, get_feature_fn)
        self.use_strong = True
        self.T = trainer_config['T']
        self.p_cutoff = trainer_config['p_cutoff']
        self.use_hard_label = trainer_config['use_hard_label']
        self.thresh_warmup = trainer_config['thresh_warmup']
    
    def train_step(self, model, img_l, target_l, class_weights, sup_loss_fn, idx_u, img_uw, img_us):
        # TODO: documentation

        # Get logits for labeled and unlabeled data.
        if not self.model_config["ret_emb"]:
            pred_l = model(img_l, ret_features=False).squeeze(-1)
            pred_uw = model(img_uw, ret_features=False).squeeze(-1)
            pred_us = model(img_us, ret_features=False).squeeze(-1)
        else:
            pred_l, _ = model(img_l, ret_features=True)
            pred_l = pred_l.squeeze(-1)
            pred_uw, _ = model(img_uw, ret_features=True)
            pred_uw = pred_uw.squeeze(-1)
            pred_us, _ = model(img_us, ret_features=True)
            pred_us = pred_us.squeeze(-1)

        # Construct supervised loss.
        if "weighted" in self.trainer_config and self.trainer_config["weighted"]:
            sup_loss = sup_loss_fn(pred_l, target_l, weight=class_weights)
        else:
            sup_loss = sup_loss_fn(pred_l, target_l)

        probs_uw = torch.softmax(pred_uw.detach())

        # TODO: get mask (see FlexMatchThresholdingHook)
        #mask =

        # TODO: get pseudo labels (PseudoLabelingHook)
        #pseudo_label = 

        # TODO: implement unsup_loss
        # unsup_loss = 

        return sup_loss + self.trainer_config["lambda_u"] * unsup_loss