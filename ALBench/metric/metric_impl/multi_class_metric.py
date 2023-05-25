import numpy as np

from ALBench.skeleton.metric_skeleton import Metric
from sklearn.metrics import f1_score

class MultiClassMetric(Metric):
    metric_name = "multi_class"

    @staticmethod
    def __accuracy(preds, targets):
        n_class = preds.shape[1]
        accs = []
        if n_class == 2:
            targets_label = targets[:, 1].astype(int)
            preds_label = (preds[:, 1] >= 0.5).astype(int)
        else:
            preds_label = np.argmax(preds, axis=-1)
            targets_label = np.argmax(targets, axis=-1)

        correct = (preds_label == targets_label).astype(float)
        count = np.sum(targets, axis=0)
        for i in range(n_class):
            target = targets[:, i]
            acc = np.sum(target * correct) / max(count[i], 1)
            accs.append(acc)
        accs = np.array(accs)

        macro_f1 = f1_score(targets_label, preds_label, average='macro')

        return np.sum(accs * count) / np.sum(count), accs, macro_f1
    

    def compute(self, epoch, preds, labels, losses, val_preds, val_labels, val_losses, test_preds, test_labels,
                test_losses, num_labeled=None, labeled=None):
        train_acc, train_accs, train_macro_f1 = self.__accuracy(preds, labels)
        val_acc, val_accs, val_macro_f1 = self.__accuracy(val_preds, val_labels)
        test_acc, test_accs, test_macro_f1 = self.__accuracy(test_preds, test_labels)
        train_loss = np.mean(losses)
        val_loss = np.mean(val_losses)
        test_loss = np.mean(test_losses)
        if labeled is not None:
            num_pos = np.sum(labels[labeled], axis=0)
        else:
            num_pos = np.zeros(labels.shape[1])
        self.num_pos = num_pos
        self.dict = {"Epoch": epoch,
                     "Num Labeled": len(preds) if num_labeled is None else num_labeled,
                     "Pool Accuracy": train_acc,
                     "Balanced Training Accuracy": np.mean(train_accs),
                     "Pool Loss": train_loss,
                     "Pool Macro F1": train_macro_f1,
                     "Validation Accuracy": val_acc,
                     "Balanced Validation Accuracy": np.mean(val_accs),
                     "Validation Loss": val_loss,
                     "Validation Macro F1": val_macro_f1,
                     "Test Accuracy": test_acc,
                     "Balanced Test Accuracy": np.mean(test_accs),
                     "Test Loss": test_loss,
                     "Test Macro F1": test_macro_f1,
                     "Total Number of Positive": np.sum(num_pos),
                     "Min Class Number of Positive": np.min(num_pos),
                     "Max Class Number of Positive": np.max(num_pos),
                     "Min Class Pool Accuracy": np.min(train_accs),
                     "Max Class Pool Accuracy": np.max(train_accs),
                     "STD Pool Accuracy": np.std(train_accs),
                     "Min Class Val Accuracy": np.min(val_accs),
                     "Max Class Val Accuracy": np.max(val_accs),
                     "STD Val Accuracy": np.std(val_accs),
                     "Min Class Test Accuracy": np.min(test_accs),
                     "Max Class Test Accuracy": np.max(test_accs),
                     "STD Test Accuracy": np.std(test_accs),
                     }
        return self.dict