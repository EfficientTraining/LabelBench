import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ALBench.trainer.trainer_impl.pytorch_passive_trainer import PyTorchPassiveTrainer
from ALBench.dataset.feature_extractor import make_semi_transforms


class PyTorchSemiTrainer(PyTorchPassiveTrainer):
    trainer_name = None

    def __init__(self, trainer_config, dataset, model_fn, model_config, metric, get_feature_fn):
        super().__init__(trainer_config, dataset, model_fn,
                         model_config, metric, get_feature_fn)
        self.use_strong = None

    def train_step(self, model, img_l, target_l, class_weights, loss_fn, idx_u, img_uw, img_us):
        raise NotImplementedError(
            "Subclass does not have implementation of semi-supervised learning training function.")

    def pretrain(self):
        pass

    def train(self, finetune_model=None, finetune_config=None):
        if self.use_strong is None:
            raise NotImplementedError("Subclass does not specify strong transformation use in use_strong.")

        model, params, loss_fn, max_epoch, optimizer, scheduler, early_stopping = self.init_train(finetune_model)

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

        # Executing pre-training (if implemented).
        self.pretrain()

        counter = 0
        for epoch in tqdm(range(max_epoch), desc="Training Epoch"):
            # For each epoch, update the embedding dataset and use the saved embedding dataset epoch.
            if "use_embeddings" in self.trainer_config and self.trainer_config["use_embeddings"]:
                self.dataset.update_embedding_dataset(epoch=epoch, get_feature_fn=self.get_feature_fn,
                                                      use_strong=self.use_strong)
                train_dataset, _, _ = self.dataset.get_embedding_datasets()

            class_weights = 1. / np.clip(np.sum(self.dataset.get_train_labels(), axis=0), a_min=1, a_max=None)
            class_weights = torch.from_numpy(class_weights).float().cuda()

            # Set dataset to return indices.
            train_dataset.set_return_indices(True)

            if epoch == 0 or ("use_embeddings" in self.trainer_config and self.trainer_config["use_embeddings"]):
                # Make labeled loader.
                labeled_dataset = Subset(train_dataset, self.dataset.labeled_idxs())
                labeled_loader = DataLoader(labeled_dataset,
                                            batch_size=self.trainer_config["train_batch_size"],
                                            shuffle=True,
                                            num_workers=self.trainer_config["num_workers"],
                                            drop_last=(len(labeled_dataset) >= self.trainer_config["train_batch_size"]))

                # Make unlabeled loader.
                unlabeled_dataset = Subset(train_dataset, self.dataset.unlabeled_idxs())
                unlabeled_loader = \
                    DataLoader(unlabeled_dataset,
                               batch_size=self.trainer_config["uratio"] * self.trainer_config["train_batch_size"],
                               shuffle=True,
                               num_workers=self.trainer_config["num_workers"],
                               drop_last=(len(unlabeled_dataset) >= self.trainer_config["train_batch_size"]))
            unlabeled_iterator = iter(unlabeled_loader)

            for img_l, target_l, *other_l in tqdm(labeled_loader, desc="Batch Index"):
                try:
                    img_u, _, idx_u = next(unlabeled_iterator)
                except StopIteration:
                    unlabeled_iterator = iter(unlabeled_loader)
                    img_u, _, idx_u = next(unlabeled_iterator)

                if self.use_strong:
                    img_uw, img_us = img_u
                    img_us = img_us.float()
                    img_l = img_l[0]
                else:
                    img_uw = img_u
                    img_us = None

                img_l, target_l = img_l.float(), target_l.float()
                img_uw = img_uw.float()

                with torch.cuda.amp.autocast():
                    loss = self.train_step(model, img_l, target_l, class_weights, loss_fn, idx_u, img_uw, img_us)

                counter = self.scheduler_step(scheduler, counter)

                optimizer.zero_grad()
                loss.backward()
                if "clip_grad" in self.trainer_config:
                    nn.utils.clip_grad_norm_(params, self.trainer_config["clip_grad"])
                optimizer.step()

            train_dataset.set_return_indices(False)

            if self.check_early_stop(early_stopping, model):
                break

        # Set embeddings back to not using semi-supervised transformation for evaluation.
        if "use_embeddings" in self.trainer_config and self.trainer_config["use_embeddings"]:
            self.dataset.update_embedding_dataset(epoch=0, get_feature_fn=self.get_feature_fn, use_strong=False)
        elif self.use_strong:
            train_dataset.set_strong_transform(None)

        return model
