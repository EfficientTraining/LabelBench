import numpy as np
import torch
from ALBench.skeleton.trainer_skeleton import Trainer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from torch.nn import functional as F
import pickle
import os


class SklearnWarmupTrainer(Trainer):
    trainer_name = "sklearn_warmup"

    def __init__(self, trainer_config, dataset, model_fn, model_config, metric, get_feature_fn):
        super().__init__(trainer_config, dataset, model_fn, model_config, metric, get_feature_fn)
        self.file_name = "sklearn_warmup_checkpoint.pickle"

    def train(self, finetune_model=None, finetune_config=None):
        train_dataset, val_dataset, test_dataset = self.dataset.get_embedding_datasets()

        if self.metric.metric_name == "multi_class":
            # Logistic regression in sklearn is for multi-class classification only.
            # Sklearn automatically adapts to binary or nonbinary classification.
            classifier = LogisticRegression(random_state=0, C=self.trainer_config["regularizer_param"],
                                            solver=self.trainer_config["optim_name"],
                                            max_iter=self.trainer_config["max_iter"], verbose=1)

            if os.path.exists(self.file_name):
                with open(self.file_name, "rb") as handle:
                    learned_coeffs = pickle.load(handle)
                classifier.coef_ = learned_coeffs["coef"]
                classifier.intercept_ = learned_coeffs["intercept"]

            # Convert one-hot encoded multi-class to the category index.
            labels = np.argmax(train_dataset.get_labels()[self.dataset.labeled_idxs()], axis=1)

            inputs = train_dataset.get_inputs()[self.dataset.labeled_idxs()]

            classifier.fit(inputs, labels)

            learned_coeffs = {}
            learned_coeffs["coef"] = classifier.coef_
            learned_coeffs["intercept"] = classifier.intercept_

            if os.path.exists(self.file_name):
                os.remove(self.file_name)
            with open(self.file_name, "wb") as handle:
                pickle.dump(learned_coeffs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            # TODO: support other metrics for sklearn training.
            raise NotImplementedError(
                "Only multi-class classification is supported for sklearn training.")

        return classifier

    def _test(self, dataset_split, model, **kwargs):

        if dataset_split == "train":
            embed_dataset = self.dataset.get_embedding_datasets()[0]
        elif dataset_split == "val":
            embed_dataset = self.dataset.get_embedding_datasets()[1]
        elif dataset_split == "test":
            embed_dataset = self.dataset.get_embedding_datasets()[2]

        preds = model.predict(embed_dataset.get_inputs())
        labels = np.argmax(embed_dataset.get_labels(), axis=1)
        loss = log_loss(labels, model.predict_proba(embed_dataset.get_inputs()))

        # Format preds and labels back to one-hot encoding.
        preds = F.one_hot(torch.from_numpy(preds), num_classes=self.dataset.num_classes).numpy()
        labels = F.one_hot(torch.from_numpy(labels), num_classes=self.dataset.num_classes).numpy()

        return preds, labels, loss, None
