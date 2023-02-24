import numpy as np

from ALBench.skeleton.metric_skeleton import Metric


class MultiLabelMetric(Metric):
    metric_name = "multi_label"

    @staticmethod
    def __average_precision(pred, target):
        indices = np.argsort(-pred)
        pos_count = 0.
        precision_at_i = 0.
        for count, i in enumerate(indices):
            label = target[i]
            if label > .5:
                pos_count += 1
                precision_at_i += pos_count / (count + 1)
        return precision_at_i / max(pos_count, 1)

    def mean_average_precision(self, preds, targets, ret_ap=False):
        n_class = preds.shape[1]
        average_precisions = []
        for i in range(n_class):
            average_precisions.append(self.__average_precision(preds[:, i], targets[:, i]))
        average_precisions = np.array(average_precisions)
        if ret_ap:
            return np.mean(average_precisions), average_precisions
        else:
            return np.mean(average_precisions)

    @staticmethod
    def __evaluate(preds, targets):
        n, n_class = preds.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            pred = preds[:, k]
            target = targets[:, k]
            target[target < .5] = 0
            Ng[k] = np.sum(target > .5)
            Np[k] = np.sum(pred >= 0.5)
            Nc[k] = np.sum(target * (pred >= 0.5))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / max(np.sum(Np), 1e-5)
        OR = np.sum(Nc) / max(np.sum(Ng), 1e-5)
        OF1 = (2 * OP * OR) / max((OP + OR), 1e-5)

        CP = np.sum(Nc / np.clip(Np, a_min=1e-5, a_max=None)) / n_class
        CR = np.sum(Nc / np.clip(Ng, a_min=1e-5, a_max=None)) / n_class
        CF1 = (2 * CP * CR) / max((CP + CR), 1e-5)
        return OP, OR, OF1, CP, CR, CF1

    @staticmethod
    def __accuracy(preds, targets):
        n_class = preds.shape[1]
        accs = []
        for i in range(n_class):
            pred = (preds[:, i] > .5)
            label = (targets[:, i] > .5)
            acc = np.mean((pred == label).astype(float))
            accs.append(acc)
        accs = np.array(accs)
        return np.mean(accs)

    def compute(self, epoch, preds, labels, losses, val_preds, val_labels, val_losses, test_preds, test_labels,
                test_losses, num_labeled=None, labeled=None):
        train_OP, train_OR, train_OF1, train_CP, train_CR, train_CF1 = self.__evaluate(preds, labels)
        val_OP, val_OR, val_OF1, val_CP, val_CR, val_CF1 = self.__evaluate(val_preds, val_labels)
        test_OP, test_OR, test_OF1, test_CP, test_CR, test_CF1 = self.__evaluate(test_preds, test_labels)
        train_acc = self.__accuracy(preds, labels)
        val_acc = self.__accuracy(val_preds, val_labels)
        test_acc = self.__accuracy(test_preds, test_labels)
        train_loss = np.mean(losses)
        val_loss = np.mean(val_losses)
        test_loss = np.mean(test_losses)
        train_map = self.mean_average_precision(preds, labels)
        val_map, val_ap = self.mean_average_precision(val_preds, val_labels, ret_ap=True)
        test_map = self.mean_average_precision(test_preds, test_labels)
        if labeled is not None:
            num_pos = np.sum(labels[labeled], axis=0)
        else:
            num_pos = np.zeros(labels.shape[1])
        self.num_pos = num_pos
        self.dict = {"Epoch": epoch,
                     "Num Labeled": len(preds) if num_labeled is None else num_labeled,
                     "Training Overall Precision": train_OP,
                     "Training Overall Recall": train_OR,
                     "Training Overall F1": train_OF1,
                     "Training Class Average Precision": train_CP,
                     "Training Class Average Recall": train_CR,
                     "Training Class Average F1": train_CF1,
                     "Validation Overall Precision": val_OP,
                     "Validation Overall Recall": val_OR,
                     "Validation Overall F1": val_OF1,
                     "Validation Class Average Precision": val_CP,
                     "Validation Class Average Recall": val_CR,
                     "Validation Class Average F1": val_CF1,
                     "Test Overall Precision": test_OP,
                     "Test Overall Recall": test_OR,
                     "Test Overall F1": test_OF1,
                     "Test Class Average Precision": test_CP,
                     "Test Class Average Recall": test_CR,
                     "Test Class Average F1": test_CF1,
                     "Training Accuracy": train_acc,
                     "Training Loss": train_loss,
                     "Training mAP": train_map,
                     "Validation Accuracy": val_acc,
                     "Validation Loss": val_loss,
                     "Validation mAP": val_map,
                     "Test Accuracy": test_acc,
                     "Test Loss": test_loss,
                     "Test mAP": test_map,
                     "Total Number of Positive": np.sum(num_pos),
                     "Min Class Number of Positive": np.min(num_pos),
                     "Max Class Number of Positive": np.max(num_pos),
                     }
        return self.dict
