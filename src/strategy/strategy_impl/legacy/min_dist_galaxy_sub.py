from src.strategy import SamplingSubProcedure
import numpy as np


class MinDistGALAXYSubProcedure(SamplingSubProcedure):
    def __init__(self, trainer, model, embs, preds, labels, labeled, dataset, batch_size, nodes):
        super(MinDistGALAXYSubProcedure, self).__init__(trainer, model, embs, preds, labels, labeled, dataset,
                                                        batch_size)
        self.init_nodes = nodes
        self.graphs = []
        for i in range(preds.shape[-1]):
            sort_idx = np.argsort(preds[:, i])
            nodes = [nodes[idx] for idx in sort_idx]
            self.graphs.append(nodes)

    def sample(self, labeled_set):
        assert len(labeled_set) > 0
        min_dist_shortest_lst = None
        min_dist_nearest_node = None
        min_dist_nearest_dist = float("inf")
        for class_idx, nodes in enumerate(self.graphs):
            last = None
            shortest_lst = None
            lst = []
            right = [None for _ in range(len(nodes))]
            left = [None for _ in range(len(nodes))]
            last_unlabel = None
            for i, node in enumerate(nodes):
                if node.labeled:
                    if (last is not None) and ((shortest_lst is None) or (len(shortest_lst) > len(lst))) and (
                            len(lst) != 0) and (node.label[class_idx] != nodes[last].label[class_idx]):
                        shortest_lst = lst
                    last = i
                    lst = []
                    next_node = nodes[min(i + 1, len(nodes) - 1)]
                    if node.label[class_idx] == 0 and next_node.labeled and next_node.label[class_idx] == 1:
                        left[i] = last_unlabel
                else:
                    lst.append(node)
                    last_unlabel = i

            last_unlabel = None
            for i in list(range(len(nodes)))[::-1]:
                cur_node = nodes[i]
                next_node = nodes[max(i - 1, 0)]
                if cur_node.labeled and next_node.labeled and (cur_node.label[class_idx] == 1) and (
                        next_node.label[class_idx] == 0):
                    right[i] = last_unlabel
                elif not cur_node.labeled:
                    last_unlabel = i
            if shortest_lst is None:
                if min_dist_shortest_lst is None:
                    nearest_node = None
                    nearest_dist = len(nodes) + 1
                    for i in range(len(nodes)):
                        if (left[i] is not None) and (nearest_dist >= i - left[i]):
                            nearest_dist = i - left[i]
                            nearest_node = left[i]
                        if (right[i] is not None) and (nearest_dist >= right[i] - i):
                            nearest_dist = right[i] - i
                            nearest_node = right[i]
                    if (nearest_node is not None) and min_dist_nearest_dist > nearest_dist:
                        min_dist_nearest_dist = nearest_dist
                        min_dist_nearest_node = nodes[nearest_node]
            elif (min_dist_shortest_lst is None) or (len(min_dist_shortest_lst) > len(shortest_lst)):
                min_dist_shortest_lst = shortest_lst

        if min_dist_shortest_lst is None:
            min_dist_nearest_node.update()
            return min_dist_nearest_node.idx
        else:
            min_dist_shortest_lst[len(min_dist_shortest_lst) // 2].update()
            return min_dist_shortest_lst[len(min_dist_shortest_lst) // 2].idx

    def update(self, idx):
        if not self.init_nodes[idx].labeled:
            self.init_nodes[idx].update()

    def __str__(self):
        return "Min Dist GALAXY"
