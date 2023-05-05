import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os

from ALBench.skeleton.trainer_skeleton import Trainer
from ALBench.trainer.utils import EarlyStopping


class PyTorchPassiveTrainer(Trainer):
    trainer_name = "pytorch_passive"

    def __init__(self, trainer_config, dataset, model_fn, model_config, metric, get_feature_fn):
        super().__init__(trainer_config, dataset, model_fn, model_config, metric, get_feature_fn)
        self.file_name = "pytorch_warmup_checkpoint.pickle"
        if self.trainer_config["warm_up"] and os.path.exists(self.file_name):
            os.remove(self.file_name)

    def train(self, finetune_model=None, finetune_config=None):

        if self.trainer_config["warm_up"] and os.path.exists(self.file_name):
            print(f"load warmup model")
            model = self.model_fn(self.model_config)
            state_dict = torch.load(self.file_name)
            remove_prefix = 'module.'
            state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in
                          state_dict.items()}
            model.load_state_dict(state_dict)
            model = model.cuda()
            model.train()
        elif finetune_model is None:
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
                train_dataset.set_transform(transform)

        # initialize the early_stopping object
        if "early_stop" in self.trainer_config and self.trainer_config["early_stop"]:
            early_stopping = EarlyStopping(
                patience=self.trainer_config["patience"] if "patience" in self.trainer_config else 5, verbose=True)
        else:
            early_stopping = None

        counter = 0
        for epoch in tqdm(range(max_epoch), desc="Training Epoch"):
            # For each epoch, update the embedding dataset and use the saved embedding dataset epoch.
            if "use_embeddings" in self.trainer_config and self.trainer_config["use_embeddings"]:
                self.dataset.update_embedding_dataset(epoch=epoch, get_feature_fn=self.get_feature_fn)
                train_dataset, _, _ = self.dataset.get_embedding_datasets()

            class_weights = 1. / np.clip(np.sum(self.dataset.get_train_labels(), axis=0), a_min=1, a_max=None)
            class_weights = torch.from_numpy(class_weights).float().cuda()
            # Only use labeled examples for training.
            train_dataset = Subset(train_dataset, self.dataset.labeled_idxs())
            loader = DataLoader(train_dataset, batch_size=self.trainer_config["train_batch_size"], shuffle=True,
                                num_workers=self.trainer_config["num_workers"],
                                drop_last=(len(train_dataset) >= self.trainer_config["train_batch_size"]))

            for img, target, *other in tqdm(loader, desc="Batch Index"):
                img, target = img.float().cuda(), target.float().cuda()

                with torch.cuda.amp.autocast():
                    if not self.model_config["ret_emb"]:
                        pred = model(img, ret_features=False).squeeze(-1)
                    else:
                        pred, _ = model(img, ret_features=True)
                        pred = pred.squeeze(-1)

                    if "weighted" in self.trainer_config and self.trainer_config["weighted"]:
                        loss = loss_fn(pred, target, weight=class_weights)
                    else:
                        loss = loss_fn(pred, target)

                if scheduler is not None:
                    scheduler(counter)
                    counter += 1
                optimizer.zero_grad()
                loss.backward()
                if "clip_grad" in self.trainer_config:
                    nn.utils.clip_grad_norm_(params, self.trainer_config["clip_grad"])
                optimizer.step()

            if early_stopping is not None:
                # Early_stopping needs the validation loss to check if it has decreased. If it has, it will make a
                # checkpoint of the current model.
                _, _, valid_losses, _ = self._test("val", model, **self.trainer_config)
                print(valid_losses.mean())
                early_stopping(valid_losses.mean(), model=None)  # Currently we don't save the model.
                if early_stopping.early_stop:
                    print("Early stopping.")
                    break
            if self.trainer_config["warm_up"] and epoch == int(max_epoch/self.trainer_config["warm_up_para"]):
                if os.path.exists(self.file_name):
                    os.remove(self.file_name)
                torch.save(model.state_dict(), self.file_name)
                print(f"save trained model")

        return model

    def _test(self, dataset_split, model, **kwargs):
        model.eval()
        if "mc_dropout" in kwargs and kwargs["mc_dropout"]:
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

        if "use_embeddings" in self.trainer_config and self.trainer_config["use_embeddings"]:
            datasets = self.dataset.get_embedding_datasets()
            assert all(dataset is not None for dataset in datasets), "Embedding features not found."
        else:
            datasets = self.dataset.get_input_datasets()

        if dataset_split == "train":
            dataset = datasets[0]
        elif dataset_split == "val":
            dataset = datasets[1]
        elif dataset_split == "test":
            dataset = datasets[2]

        # If clip model, we need to update the transform of dataset.
        if "use_customized_transform" in self.model_config and self.model_config["use_customized_transform"]:
            transform = model.module.get_preprocess(split="test")
            dataset.set_transform(transform)

        loader = DataLoader(dataset, batch_size=self.trainer_config["test_batch_size"], shuffle=False, num_workers=10)
        preds = np.zeros((len(dataset), self.dataset.num_classes), dtype=float)
        labels = np.zeros((len(dataset), self.dataset.num_classes), dtype=float)
        losses = np.zeros(len(dataset), dtype=float)
        embs = []
        counter = 0
        for img, target in loader:
            img, target = img.float().cuda(), target.float().cuda()
            with torch.no_grad(), torch.cuda.amp.autocast():
                if not self.model_config["ret_emb"]:
                    pred = model(img)
                else:
                    pred, emb = model(img, ret_features=True)
                    embs.append(emb.cpu().numpy())
                pred = pred.squeeze(-1)
                loss = self.trainer_config["loss_fn"](pred, target)
            preds[counter: (counter + len(pred))] = self.trainer_config["pred_fn"](pred).cpu().numpy()
            labels[counter: (counter + len(pred))] = target.cpu().numpy()
            losses[counter: (counter + len(pred))] = loss.cpu().numpy()
            counter += len(pred)
        assert counter == len(preds)
        model.train()
        return preds, labels, losses, np.concatenate(embs, axis=0) if len(embs) > 0 else None
