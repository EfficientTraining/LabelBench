import torch

from ..pytorch_semi_trainer import PyTorchSemiTrainer # TODO: is this right?

class FlexmatchTrainer(PyTorchSemiTrainer):
    trainer_name = "flexmatch" # TODO: is this being set correctly, will the parent trainer_name be overridden?

    def __init__(self, trainer_config, dataset, model_fn, model_config, metric, get_feature_fn):
        super().__init__(trainer_config, dataset, model_fn, model_config, metric, get_feature_fn)
        self.use_strong = True
    
    def train_step(self, img_l, target_l, img_uw, img_us=None):
        # TODO: implement flexmatch here, return loss

        with torch.cuda.amp.autocast():
            pass
            #if not self.model_config["ret_emb"]:
            #    pred = model(img, ret_features=False).squeeze(-1)
            #else:
            #    pred, _ = model(img, ret_features=True)
            #    pred = pred.squeeze(-1)

            #if "weighted" in self.trainer_config and self.trainer_config["weighted"]:
            #    loss = loss_fn(pred, target, weight=class_weights)
            #else:
            #    loss = loss_fn(pred, target)

        loss = None
        return loss