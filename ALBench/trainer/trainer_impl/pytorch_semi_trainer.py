import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchvision import transforms

from pytorch_passive_trainer import PyTorchPassiveTrainer
from ALBench.trainer.utils import EarlyStopping
from ALBench.trainer.trainer_impl.semi_algs.randaugment import RandAugment

def make_semi_transforms(transform):
    # TODO: for now, just returning transform_weak as the original transform itself. Can inject additional randomness later if needed
    transform_weak = transform

    # instantiate transform_strong as a copy of transform
    transform_strong = copy.deepcopy(transform)

    # FIXME: this will break if transform is just a single transform instead of a composition. also assumes that ToTensor is in the transform
    for i, t in enumerate(transform_strong.transforms):
        if isinstance(t, transforms.ToTensor):
            # based on strong augmentation used in https://github.com/microsoft/Semi-supervised-learning
            transform_strong.transforms.insert(i-1, RandAugment(3, 5))

    return transform_weak, transform_strong


class PyTorchSemiTrainer(PyTorchPassiveTrainer):
    trainer_name = "pytorch_semi" # TODO: is this being set correctly. Consider setting at flexmatch level, not here at the semi base

    def __init__(self, trainer_config, dataset, model_fn, model_config, metric, get_feature_fn):
        super().__init__(trainer_config, dataset, model_fn, model_config, metric, get_feature_fn)
        self.use_strong = None # (bool) flag if semi-supervised method needs strong data augmentation. Must be implemented by subclass
    
    def train_step(self, img_l, target_l, img_u, target_u):
        raise NotImplementedError("Subclass does not have implementation of semi-supervised learning training function.")

    def train(self, finetune_model=None, finetune_config=None):

        if self.use_strong is None:
            raise NotImplementedError("Subclass does not specify strong transformation use in use_strong.")
        
        if finetune_model is None:
            model = self.model_fn(self.model_config)
            if "template" in self.model_config:
                if hasattr(model, 'init_head_withzeroshot'):
                    model.init_head_withzeroshot(classnames=self.dataset.get_classnames(),
                                                 template=self.model_config["template"])
                else:
                    raise ValueError("Please use a model that supports zero-shot learning.")
            model = model.cuda()
        else:
            model = copy.deepcopy(finetune_model).cuda()

        loss_fn = self.trainer_config["loss_fn"]
        max_epoch = self.trainer_config["max_epoch"]

        devices = list(range(torch.cuda.device_count()))
        print('Using devices', devices)
        model = torch.nn.parallel.DataParallel(model, device_ids=devices)

        params = [p for p in model.parameters() if p.requires_grad]

        optimizer = self.trainer_config["optim_fn"](params)
        total_steps = self.trainer_config["max_epoch"] * len(self.dataset) // self.trainer_config["train_batch_size"]
        scheduler = self.trainer_config["scheduler_fn"](optimizer, total_steps) \
            if "scheduler_fn" in self.trainer_config else None

        # Check to avoid using customized transform for embedding dataset.
        if "use_customized_transform" in self.model_config and "use_embeddings" in self.trainer_config:
            assert not (self.trainer_config["use_embeddings"] and self.model_config["use_customized_transform"]), \
                "Customized transform is only supported for non-embedding models."

        # Get the training dataset for the non-embedding dataset.
        if "use_embeddings" not in self.trainer_config or (not self.trainer_config["use_embeddings"]):
            train_dataset, _, _ = self.dataset.get_input_datasets()
            if "use_customized_transform" in self.model_config and self.model_config["use_customized_transform"]:
                transform = model.module.get_preprocess(split="train")
            else:
                transform = train_dataset.get_transform()

            transform_weak, transform_strong = make_semi_transforms(transform)
            train_dataset.set_transform(transform_weak)
            if self.use_strong:
                assert transform_strong is not None, "Strong transform cannot be None"
                train_dataset.set_strong_transform(transform_strong)

        # initialize the early_stopping object
        early_stopping = EarlyStopping(
            patience=self.trainer_config["patience"] if "patience" in self.trainer_config else None, verbose=True)

        counter = 0
        for epoch in tqdm(range(max_epoch), desc="Training Epoch"):
            # For each epoch, update the embedding dataset and use the saved embedding dataset epoch.
            if "use_embeddings" in self.trainer_config and self.trainer_config["use_embeddings"]:

                # FIXME:
                raise ValueError("use_embeddings not yet implemented for semiSL.")
                #self.dataset.update_embedding_dataset(epoch=epoch, get_feature_fn=self.get_feature_fn)
                #train_dataset, _, _ = self.dataset.get_embedding_datasets()

            class_weights = 1. / np.clip(np.sum(self.dataset.get_train_labels(), axis=0), a_min=1, a_max=None)
            class_weights = torch.from_numpy(class_weights).float().cuda()

            # Make labeled loader.
            labeled_dataset = Subset(train_dataset, self.dataset.labeled_idxs())
            labeled_loader = DataLoader(labeled_dataset, batch_size=self.trainer_config["train_batch_size"], shuffle=True,
                                num_workers=self.trainer_config["num_workers"],
                                drop_last=(len(labeled_dataset) >= self.trainer_config["train_batch_size"]))
            
            # Make unlabeled loader.
            unlabeled_dataset = Subset(train_dataset, self.dataset.unlabeled_idxs())
            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=self.trainer_config["uratio"]*
                                self.trainer_config["train_batch_size"], shuffle=True,
                                num_workers=self.trainer_config["num_workers"],
                                drop_last=False)
            unlabeled_iterator = iter(unlabeled_loader) # need to manually mix in unlabeled batches

            for img_l, target_l, *other_l in tqdm(labeled_loader, desc="Batch Index"):

                # Get unlabeled batch.
                # https://stackoverflow.com/questions/51444059/how-to-iterate-over-two-dataloaders-simultaneously-using-pytorch
                # https://github.com/pytorch/pytorch/issues/1917#issuecomment-433698337
                # Alternative is to zip loaders as in:
                #     https://github.com/microsoft/Semi-supervised-learning/blob/
                #     9a24e00db040443b4b8d13ecb556b6948c56d15e/semilearn/core/algorithmbase.py#L301
                # But zipping stops at the shortest loader, which can create problems towards the end of active training
                # when unlabeled set is smaller than labeled set

                try:
                    img_uw, _, *other_uw = next(unlabeled_iterator)
                except StopIteration:
                    unlabeled_iterator = iter(unlabeled_loader)
                    img_uw, _, *other_uw = next(unlabeled_iterator)

                img_l, target_l = img_l.float().cuda(), target_l.float().cuda()
                img_uw = img_uw.float().cuda()

                if len(other_uw) > 0:
                    img_us = other_uw[0].float.cuda()
                else:
                    img_us = None

                with torch.cuda.amp.autocast():
                    # TODO: need unlabeled indices idx_u
                    loss = self.train_step(model, img_l, target_l, class_weights, loss_fn, idx_u, img_uw, img_us)

                if scheduler is not None:
                    scheduler(counter)
                    counter += 1
                optimizer.zero_grad()
                loss.backward()
                if "clip_grad" in self.trainer_config:
                    nn.utils.clip_grad_norm_(params, self.trainer_config["clip_grad"])
                optimizer.step()

            if "early_stop" in self.trainer_config and self.trainer_config["early_stop"]:
                # Early_stopping needs the validation loss to check if it has decreased. If it has, it will make a
                # checkpoint of the current model.
                _, _, valid_losses, _ = self._test("val", model, **self.trainer_config)
                print(valid_losses.mean())
                early_stopping(valid_losses.mean(), model=None)  # Currently we don't save the model.
                if early_stopping.early_stop:
                    print("Early stopping.")
                    break
        return model