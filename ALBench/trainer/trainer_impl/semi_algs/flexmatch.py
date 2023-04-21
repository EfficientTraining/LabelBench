from ../pytorch_semi_trainer import PyTorchSemiTrainer # FIXME: this isn't right, come back to this

class FlexmatchTrainer(PyTorchSemiTrainer):
    trainer_name = "flexmatch" # TODO: is this being set correctly, will the parent trainer_name be overridden?

    def __init__(self, trainer_config, dataset, model_fn, model_config, metric, get_feature_fn):
        super().__init__(trainer_config, dataset, model_fn, model_config, metric, get_feature_fn)
    
    def train_step(self, img_l, target_l, img_u, target_u):
        # TODO: implement flexmatch here, return loss
        loss = None
        return loss