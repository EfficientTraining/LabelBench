import math
import numpy as np
from tqdm import tqdm

from LabelBench.skeleton.active_learning_skeleton import Strategy, ALInput


class Node:
    def __init__(self, idx, pred, label):
        self.idx = idx
        self.pred = pred
        self.labeled = False
        self.label = label

    def update(self):
        assert not self.labeled
        self.labeled = True


class DIRECTSampling(Strategy):
    strategy_name = "direct"

    def __init__(self, strategy_config, dataset):
        super(DIRECTSampling, self).__init__(strategy_config, dataset)
        self.input_types = [ALInput.TRAIN_PRED, ALInput.TRAIN_LABEL]

    @staticmethod
    def get_acc(w):
        cumsum = np.insert(np.cumsum(w), 0, 0)
        reverse_cumsum = np.insert(np.cumsum(w[::-1]), 0, 0)[::-1]
        # Obtain accuracy vector
        acc = cumsum - reverse_cumsum
        print("acc", acc)
        return acc

    def find_best_hyp(self, w, n):
        acc = self.get_acc(w)
        best_hyp = np.argmax(acc)
        print("best_hyp", best_hyp)
        return best_hyp

    def version_space_reduction(self, graph, query_idxs, I, J, B1, B2, n, w, k, spend):
        print("Begin version space reduction")
        print("I: %d =, J: %d, B1: %d, B2: %d, spend:%d" % (I, J, B1, B2, spend))

        # BASE CASE: return total amount spent
        if B2 == 0:
            return spend, query_idxs

        # RECURSIVE CASE: narrow down version space
        budget = min(B1, B2)
        # Construct lambda
        lmbda = []
        for _ in range(I):
            lmbda.append(0)
        for _ in range(I, J + 1):
            lmbda.append(1 / (J - I + 1))
        for _ in range(J + 1, n):
            lmbda.append(0)
        lmbda = np.array(lmbda)
        # Sample according to lambda without replacement.
        # Step 1: Create a new lambda vector which will reflect probabilities for only unlabeled indices.
        lmbda_unlabeled = []
        for i, node in enumerate(graph):
            if node.labeled:
                lmbda_unlabeled.append(0)
            else:
                lmbda_unlabeled.append(lmbda[i])
        # Step 2: Sample according to lmbda_unlabeled
        # Step 2.1: If every lambda is 0, terminate by returning spend
        if sum(lmbda_unlabeled) == 0:
            return spend, query_idxs
        # Step 2.2: Sample budget or if budget is too large, sample all nonzero indices
        greater_than_zero = [num for num in lmbda_unlabeled if num > 0]
        num_sample = min(budget, len(greater_than_zero))
        samp_idxs = np.random.choice(n, size=num_sample, replace=False,
                                     p=np.array(lmbda_unlabeled / sum(lmbda_unlabeled)))

        # Step 3: Update query_idxs and spend tracker
        for idx in samp_idxs:
            assert not graph[idx].labeled
            query_idxs.add(graph[idx].idx)
            print("samp idx: ", idx)
            spend += 1

        # Step 4: Label every samp_idx and update w
        for i in samp_idxs:
            graph[i].update()
            if np.argmax(graph[i].label) == k:
                w[i] = 1
            else:
                w[i] = -1

        # Step 5: Compute diameter of uncertainty region. We only count unlabeled examples
        cur_diam = 0
        for m in range(I, J + 1):
            if not graph[m].labeled:
                cur_diam += 1
        print("cur_diam: ", cur_diam)

        # Shrink version space
        if cur_diam > 0:
            # Step 2: Compute shrinkage factor
            base = B2 // B1
            if base <= 1:
                c2 = 1
            else:
                # c1 = math.log(cur_diam, B2 // B1 + 1)
                c2 = cur_diam ** (1 / (B2 // B1))
            # print("c1: ", c1)
            print("c2: ", c2)
            # Step 3: Iteratively shrink version space
            diam = cur_diam
            print("target diam: ", math.ceil(cur_diam / c2))
            while diam > math.ceil(cur_diam / c2):
                # Compare accuracy of left and right hypothesis
                accs = self.get_acc(w)
                # Step 3.3: Determine which accuracy is higher. Shift the pointer of the lower accuracy hypothesis
                if accs[I] < accs[J]:
                    if not graph[I].labeled:
                        diam -= 1
                    I += 1
                elif accs[I] > accs[J]:
                    if not graph[J].labeled:
                        diam -= 1
                    J -= 1
                else:
                    b = np.random.randint(0, 2)
                    if b == 0:
                        if not graph[I].labeled:
                            diam -= 1
                        I += 1
                    else:
                        if not graph[J].labeled:
                            diam -= 1
                        J -= 1
            print("diam:", diam)

        return self.version_space_reduction(graph, query_idxs, I, J, B1, B2 - budget, n, w, k, spend)

    def select(self, trainer, budget):
        labeled_set = set(list(self.dataset.labeled_idxs()))
        assert len(labeled_set) > 0

        preds, labels = trainer.retrieve_inputs(self.input_types)
        n_class = preds.shape[1]
        nodes = []
        most_confident = np.max(preds, axis=1).reshape((-1, 1))
        margins = preds - most_confident + 1e-8 * (most_confident if n_class > 2 else preds[:, 0:1])
        for idx, (margin, label) in enumerate(zip(margins, labels)):
            nodes.append(Node(idx, margin, label))
        class_freq = np.zeros(n_class)
        for i in labeled_set:
            nodes[i].update()
            class_freq[np.argmax(nodes[i].label)] += 1

        # List of nodes separated by class where each list of nodes is sorted by confidence score
        graphs = []
        reverse_graphs = []
        for i in range(n_class):
            # Sort nodes in every class by margin.
            sort_idx = np.argsort(-margins[:, i])
            graphs.append([nodes[idx] for idx in sort_idx])
            reverse_graphs.append([nodes[idx] for idx in sort_idx[::-1]])

        # This is what we return. This will be the additional indices that we query on this iteration.
        query_idxs = set()
        # This tracks how much we have spent on version space reduction.
        version_spend = 0
        # Number of parallel annotators.
        B1 = self.strategy_config["B1"]
        class_rank = np.argsort(-class_freq)

        # Batch size per class
        if budget // n_class // 2 < self.strategy_config["min_budget_per_class"]:
            eff_class = n_class
            B2 = budget // n_class // 2
        else:
            eff_class = budget // (2 * self.strategy_config["min_budget_per_class"])
            B2 = self.strategy_config["min_budget_per_class"]

        # VERSION SPACE REDUCTION
        # Start collect examples and label.
        for i, k in enumerate(class_rank[-eff_class:]):
            graph = graphs[k]
            reverse_graph = reverse_graphs[k]

            # Initialize w
            w = []
            for node in graph:
                if not node.labeled:
                    w.append(0)
                elif np.argmax(node.label) == k:
                    w.append(1)
                else:
                    w.append(-1)

            # Find region of uncertainty and perform version reduction
            # Step 1: Find I*
            I = 0
            for idx, node in enumerate(graph):
                # I* must be labeled
                if not node.labeled:
                    continue
                # I* is found
                if np.argmax(node.label) != k:
                    break
                I = idx
            # Step 2: Find J*
            J = 0
            for idx, node in enumerate(reverse_graph):
                # J* must be labeled
                if not node.labeled:
                    continue
                # J* is found
                if np.argmax(node.label) == k:
                    break
                J = idx
            # Step 3: Update J to due to prior reversal of graph
            J = len(graph) - J - 1
            # Step 4: Update B2 if there are too few unlabeled examples
            unlabeled = 0
            for node in graph:
                if not node.labeled:
                    unlabeled += 1
            B2 = min(B2, unlabeled)
            # Step 5: Spend half of budget reducing the version space.
            spend = 0
            spend, query_idxs = self.version_space_reduction(graph, query_idxs, I, J, B1, B2, len(graph), np.array(w),
                                                             k, spend)
            version_spend += spend
        print("Total spent in version space reduction: ", version_spend)

        """QUERY AROUND HYPOTHESIS"""
        rem_budget = budget - version_spend
        print("remaining budget: ", rem_budget)

        for i, k in enumerate(np.random.permutation(n_class)):
            graph = graphs[k]

            # Initialize w
            w = []
            for node in graph:
                if not node.labeled:
                    w.append(0)
                elif np.argmax(node.label) == k:
                    w.append(1)
                else:
                    w.append(-1)

            # Find best hypothesis
            best_hyp = self.find_best_hyp(w, len(graph))

            # Query around this hypothesis
            right = best_hyp
            left = right - 1
            bit = 1
            left_exceed, right_exceed = False, False

            # Step 2: Keep moving away from point and query labels
            class_budget = rem_budget * (i + 1) // n_class - rem_budget * i // n_class
            while class_budget > 0:
                # Update bit
                bit = (bit + 1) % 2
                # We have labeled as much as we can; break
                if left_exceed and right_exceed:
                    break
                # Label to left
                if bit == 0:
                    if left < 0:
                        left_exceed = True
                        continue
                    if not graph[left].labeled:
                        query_idxs.add(graph[left].idx)
                        graph[left].update()
                        class_budget -= 1
                    left -= 1
                # Label to right
                else:
                    if right >= len(labels):
                        right_exceed = True
                        continue
                    if not graph[right].labeled:
                        query_idxs.add(graph[right].idx)
                        graph[right].update()
                        class_budget -= 1
                    right += 1

        return list(query_idxs)
