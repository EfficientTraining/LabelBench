from src.strategy import SamplingSubProcedure
import numpy as np


class Node:
    def __init__(self, idx, pred, label):
        self.idx = idx
        self.pred = pred
        self.labeled = False
        self.label = label

    def update(self):
        assert not self.labeled
        self.labeled = True


class GALAXYSubProcedure(SamplingSubProcedure):
    def __init__(self, trainer, model, embs, preds, labels, labeled, dataset, batch_size, nodes, class_idx):
        super(GALAXYSubProcedure, self).__init__(trainer, model, embs, preds, labels, labeled, dataset, batch_size)
        sort_idx = np.argsort(preds)
        self.init_nodes = nodes
        nodes = [nodes[idx] for idx in sort_idx]
        self.idx = 0
        self.nodes = nodes
        self.class_idx = class_idx

    def sample(self, labeled_set):
        assert len(labeled_set) > 0
        last = None
        shortest_lst = None
        lst = []
        right = [None for _ in range(len(self.nodes))]
        left = [None for _ in range(len(self.nodes))]
        last_unlabel = None
        for i, node in enumerate(self.nodes):
            if node.labeled:
                if (last is not None) and ((shortest_lst is None) or (len(shortest_lst) > len(lst))) and (
                        len(lst) != 0) and (node.label[self.class_idx] != self.nodes[last].label[self.class_idx]):
                    shortest_lst = lst
                last = i
                lst = []
                next_node = self.nodes[min(i + 1, len(self.nodes) - 1)]
                if node.label[self.class_idx] == 0 and next_node.labeled and next_node.label[self.class_idx] == 1:
                    left[i] = last_unlabel
            else:
                lst.append(node)
                last_unlabel = i

        last_unlabel = None
        for i in list(range(len(self.nodes)))[::-1]:
            cur_node = self.nodes[i]
            next_node = self.nodes[max(i - 1, 0)]
            if cur_node.labeled and next_node.labeled and (cur_node.label[self.class_idx] == 1) and (
                    next_node.label[self.class_idx] == 0):
                right[i] = last_unlabel
            elif not cur_node.labeled:
                last_unlabel = i

        if shortest_lst is None:
            nearest_node = None
            nearest_dist = len(self.nodes) + 1
            for i in range(len(self.nodes)):
                if (left[i] is not None) and (nearest_dist >= i - left[i]):
                    nearest_dist = i - left[i]
                    nearest_node = left[i]
                if (right[i] is not None) and (nearest_dist >= right[i] - i):
                    nearest_dist = right[i] - i
                    nearest_node = right[i]
            # If one or two classes don't have labeled node, randomly choose one.
            if nearest_node is None:
                perm = np.random.permutation(len(self.nodes))
                for i in perm:
                    if not self.nodes[i].labeled:
                        nearest_node = i
                        break
            self.nodes[nearest_node].update()
            return self.nodes[nearest_node].idx
        else:
            shortest_lst[len(shortest_lst) // 2].update()
            return shortest_lst[len(shortest_lst) // 2].idx

    def update(self, idx):
        if not self.init_nodes[idx].labeled:
            self.init_nodes[idx].update()

    def __str__(self):
        return "GALAXY, Class #%d" % self.class_idx
