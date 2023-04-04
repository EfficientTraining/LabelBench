import numpy as np
from tqdm import tqdm

from ALBench.skeleton.active_learning_skeleton import Strategy, ALInput


class Node:
    def __init__(self, idx, pred, label):
        self.idx = idx
        self.pred = pred
        self.labeled = False
        self.label = label

    def update(self):
        assert not self.labeled
        self.labeled = True


class GALAXYSampling(Strategy):
    strategy_name = "galaxy"

    def __init__(self, strategy_config, dataset):
        super(GALAXYSampling, self).__init__(strategy_config, dataset)
        self.input_types = {ALInput.TRAIN_PRED, ALInput.TRAIN_LABEL}

    def select(self, trainer, budget):
        labeled_set = set(list(self.dataset.labeled_idxs()))
        assert len(labeled_set) > 0

        preds, labels = trainer.retrieve_inputs(self.input_types)
        n_class = preds.shape[1]
        nodes = []
        most_confident = np.max(preds, axis=1).reshape((-1, 1))
        margins = preds - most_confident + 1e-8 * most_confident if n_class > 2 else preds[:, 0:1]
        for idx, (margin, label) in enumerate(zip(margins, labels)):
            nodes.append(Node(idx, margin, label))
        for i in labeled_set:
            nodes[i].update()

        graphs = []
        graphs_left = []
        graphs_right = []
        for i in range(n_class if n_class > 2 else 1):
            sort_idx = np.argsort(-margins[:, i])
            graphs.append([nodes[idx] for idx in sort_idx])
            graphs_left.append(graphs[-1][1:] + [graphs[-1][-1]])
            graphs_right.append(([graphs[-1][0]] + graphs[-1][:-1])[::-1])

        # Start collect examples and label.
        query_idxs = []
        for _ in tqdm(range(budget)):
            k = np.random.randint(n_class) if n_class > 2 else 0
            graph = graphs[k]
            graph_left = graphs_left[k]
            graph_right = graphs_right[k]
            last = None
            shortest_lst = None
            lst = None
            last_unlabel_left = None
            last_unlabel_right = None
            nearest_node = None
            nearest_dist = len(graph) + 1

            # Left to right.
            node = graph[0]
            for i, next_node_left in enumerate(graph_left):
                if node.labeled:
                    if (last is not None) and (lst is not None) and \
                            ((shortest_lst is None) or (shortest_lst[1] - shortest_lst[0] > lst[1] - lst[0])) and \
                            (node.label[k] != last.label[k]):
                        shortest_lst = lst
                    last = node
                    lst = None
                    if node.label[k] == 0 and next_node_left.labeled and next_node_left.label[k] == 1 and \
                            last_unlabel_left is not None and nearest_dist >= i - last_unlabel_left:
                        nearest_dist = i - last_unlabel_left
                        nearest_node = last_unlabel_left
                else:
                    if lst is None:
                        lst = [i, i]
                    lst[1] = i
                    last_unlabel_left = i
                node = next_node_left

            if shortest_lst is None:
                # Right to left.
                node = graph[-1]
                for i, next_node_right in enumerate(graph_right):
                    i_right = len(graph) - 1 - i
                    if node.labeled and next_node_right.labeled and node.label[k] == 1 and \
                            next_node_right.label[k] == 0 and last_unlabel_right is not None and \
                            nearest_dist > last_unlabel_right - i_right:
                        nearest_dist = last_unlabel_right - i_right
                        nearest_node = last_unlabel_right
                    elif not node.labeled:
                        last_unlabel_right = i_right
                    node = next_node_right

                # If one or two classes don't have labeled node, randomly choose one.
                if nearest_node is None:
                    perm = np.random.permutation(len(graph))
                    for i in perm:
                        if not graph[i].labeled:
                            nearest_node = i
                            break
                graph[nearest_node].update()
                query_idxs.append(graph[nearest_node].idx)
            else:
                idx = (shortest_lst[0] + shortest_lst[1]) // 2
                graph[idx].update()
                query_idxs.append(graph[idx].idx)

        return query_idxs
