import copy

import torch.nn.functional as F
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.skeleton.trainer_skeleton import Trainer


class PassiveTrainer(Trainer):
    trainer_name = "passive"

    def __init__(self, trainer_config, dataset, model_fn, metric):
        super().__init__(trainer_config, dataset, model_fn, metric)
        # self.batch_size = batch_size
        # self.model_class = model_class
        # self.n_class = n_class
        # self.n_epochs = n_epochs
        # self.loss_fn = loss_fn
        # self.metric = metric
        # self.pred_fn = pred_fn
        # self.weighted = False
        # self.multi_label_flag = multi_label_flag

    def train(self, train_dataset, test_dataset, finetune=(None, None, None), log=False, lr=1e-4):
        model, loss_fn, n_epoch = finetune
        if model is None:
            model = self.model_class(self.n_class).cuda()
        else:
            model = copy.deepcopy(model).cuda()
        if loss_fn is None:
            loss_fn = self.loss_fn
        if n_epoch is None:
            n_epoch = self.n_epochs
        model.train()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-5)
        loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10)

        for epoch in range(n_epoch):
            preds = np.zeros((len(train_dataset), self.n_class), dtype=float)
            labels = np.zeros((len(train_dataset), self.n_class), dtype=float)
            losses = np.zeros(len(train_dataset), dtype=float)
            counter = 0
            for img, target, *other in loader:
                img, target = img.float().cuda(), target.float().cuda()
                pred = model(img).squeeze(-1)
                loss = loss_fn(pred, target, *other)

                preds[counter: (counter + len(pred))] = self.pred_fn(pred.data).cpu().numpy()
                labels[counter: (counter + len(pred))] = target.data.cpu().numpy()
                losses[counter: (counter + len(pred))] = loss.data.cpu().numpy()
                counter += len(pred)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if log:
                test_preds, test_labels, test_losses = self.test(test_dataset, model)
                log_dict = self.metric.compute(epoch, preds, labels, losses, test_preds, test_labels, test_losses,
                                               test_preds, test_labels, test_losses)
                wandb.log(log_dict)
        return model

    def test(self, dataset, model, ret_emb=False):
        model.eval()
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False, num_workers=10)
        preds = np.zeros((len(dataset), self.n_class), dtype=float)
        labels = np.zeros((len(dataset), self.n_class), dtype=float)
        losses = np.zeros(len(dataset), dtype=float)
        embs = []
        counter = 0
        for img, target in loader:
            img, target = img.float().cuda(), target.float().cuda()
            with torch.no_grad():
                if ret_emb:
                    pred, emb = model(img, ret_features=True)
                    pred, emb = pred.squeeze(-1), emb
                    embs.append(emb.cpu().numpy())
                else:
                    pred = model(img).squeeze(-1)
                loss = self.loss_fn(pred, target)
            preds[counter: (counter + len(pred))] = self.pred_fn(pred).cpu().numpy()
            labels[counter: (counter + len(pred))] = target.cpu().numpy()
            losses[counter: (counter + len(pred))] = loss.cpu().numpy()
            counter += len(pred)
        assert counter == len(preds)
        model.train()
        if ret_emb:
            return preds, labels, losses, np.concatenate(embs, axis=0)
        else:
            return preds, labels, losses
