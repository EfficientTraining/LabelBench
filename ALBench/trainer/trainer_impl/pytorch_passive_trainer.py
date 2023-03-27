import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from ALBench.skeleton.trainer_skeleton import Trainer


class PyTorchPassiveTrainer(Trainer):
    trainer_name = "pytorch_passive"

    def __init__(self, trainer_config, dataset, model_fn, model_config, metric, input_dim):
        super().__init__(trainer_config, dataset, model_fn, model_config, metric, input_dim)

    def train(self, finetune_model=None, finetune_config=None):
        if finetune_model is None:
            model = self.model_fn(self.model_config, self.input_dim).cuda()
        else:
            model = copy.deepcopy(finetune_model).cuda()

        loss_fn = self.trainer_config["loss_fn"]
        max_epoch = self.trainer_config["max_epoch"]
        model.train()
        optimizer = self.trainer_config["optim_fn"](model.parameters())

        if self.trainer_config["use_embeddings"]:
            train_dataset, val_dataset, test_dataset = self.dataset.get_embedding_datasets()
        else:
            train_dataset, val_dataset, test_dataset = self.dataset.get_input_datasets()

        loader = DataLoader(train_dataset, batch_size=self.trainer_config["train_batch_size"], shuffle=True, num_workers=self.trainer_config["num_workers"])

        for epoch in range(max_epoch):
            preds = np.zeros((len(train_dataset), self.dataset.num_classes), dtype=float)
            labels = np.zeros((len(train_dataset), self.dataset.num_classes), dtype=float)
            losses = np.zeros(len(train_dataset), dtype=float)
            counter = 0
            for img, target, *other in loader:
                img, target = img.float().cuda(), target.float().cuda()

                with torch.cuda.amp.autocast():
                    if not self.model_config["ret_emb"]:
                        pred = model(img, ret_features=False).squeeze(-1)
                    else:
                        pred, _ = model(img, ret_features=True)
                        pred = pred.squeeze(-1)
                    loss = loss_fn(pred, target, *other)

                preds[counter: (counter + len(pred))] = self.trainer_config["pred_fn"](pred.data).cpu().numpy()
                labels[counter: (counter + len(pred))] = target.data.cpu().numpy()
                losses[counter: (counter + len(pred))] = loss.data.cpu().numpy()
                counter += len(pred)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model

    def _test(self, dataset_split, model, **kwargs):
        model.eval()
        if "mc_dropout" in kwargs and kwargs["mc_dropout"]:
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

        if self.trainer_config["use_embeddings"]:
            datasets = self.dataset.get_embedding_datasets()
            assert all(
                dataset is not None for dataset in datasets), "embedding features not found"
        else:
            datasets = self.dataset.get_input_datasets()

        if dataset_split == "train":
            dataset = datasets[0]
        elif dataset_split == "val":
            dataset = datasets[1]
        elif dataset_split == "test":
            dataset = datasets[2]

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


