from src.strategy import SamplingSubProcedure
import submodlib
import numpy as np


def get_balanced_set(labeled_set, labels, n_class, k):
    labels = np.argmax(labels, axis=-1)
    lists = [[] for _ in range(n_class)]
    for idx in labeled_set:
        label = labels[idx]
        if len(lists[label]) < k:
            lists[label].append(idx)
    flatten_lst = []
    for lst in lists:
        flatten_lst = flatten_lst + lst
    lists = np.array(flatten_lst, dtype=int)
    return lists


class SMISubProcedure(SamplingSubProcedure):
    def __init__(self, trainer, model, embs, preds, labels, labeled, dataset, batch_size, args, grad_embeddings,
                 unlabeled, labeled_subset):
        super(SMISubProcedure, self).__init__(trainer, model, embs, preds, labels, labeled, dataset, batch_size)
        self.args = args
        if np.min(np.sum(labels[labeled], axis=-1)) < 5:
            labeled_set = set(list(labeled))
            all_set = set(list(range(len(dataset))))
            unlabeled = np.array(list(all_set - labeled_set))
            self.top_idxs = np.random.choice(unlabeled, size=batch_size, replace=False)
        else:
            assert grad_embeddings.shape[0] == len(unlabeled) + len(labeled_subset)
            # Get hyperparameters from args dict
            optimizer = self.args['optimizer'] if 'optimizer' in self.args else 'NaiveGreedy'
            metric = self.args['metric'] if 'metric' in self.args else 'cosine'
            eta = self.args['eta'] if 'eta' in self.args else 1
            stopIfZeroGain = self.args['stopIfZeroGain'] if 'stopIfZeroGain' in self.args else False
            stopIfNegativeGain = self.args['stopIfNegativeGain'] if 'stopIfNegativeGain' in self.args else False
            verbose = self.args['verbose'] if 'verbose' in self.args else False
            embedding_type = self.args['embedding_type'] if 'embedding_type' in self.args else "gradients"

            # Compute Embeddings
            if embedding_type == "gradients":
                unlabeled_data_embedding = grad_embeddings[:len(unlabeled)].reshape((len(unlabeled), -1))
                query_embedding = grad_embeddings[len(unlabeled):].reshape((len(labeled_subset), -1))
            elif embedding_type == "features":
                unlabeled_data_embedding = embs[unlabeled]
                query_embedding = embs[labeled_subset]
            else:
                raise ValueError("Provided representation must be one of gradients or features")

            # Compute image-image kernel
            if self.args['smi_function'] == 'fl1mi' or self.args['smi_function'] == 'logdetmi':
                data_sijs = submodlib.helper.create_kernel(X=unlabeled_data_embedding, metric=metric, method="sklearn")
            # Compute query-query kernel
            if self.args['smi_function'] == 'logdetmi':
                query_query_sijs = submodlib.helper.create_kernel(X=query_embedding, metric=metric, method="sklearn")
            # Compute image-query kernel
            query_sijs = submodlib.helper.create_kernel(X=query_embedding, X_rep=unlabeled_data_embedding,
                                                        metric=metric,
                                                        method="sklearn")

            if self.args['smi_function'] == 'fl1mi':
                obj = submodlib.FacilityLocationMutualInformationFunction(n=unlabeled_data_embedding.shape[0],
                                                                          num_queries=query_embedding.shape[0],
                                                                          data_sijs=data_sijs,
                                                                          query_sijs=query_sijs,
                                                                          magnificationEta=eta)

            if self.args['smi_function'] == 'fl2mi':
                obj = submodlib.FacilityLocationVariantMutualInformationFunction(n=unlabeled_data_embedding.shape[0],
                                                                                 num_queries=query_embedding.shape[0],
                                                                                 query_sijs=query_sijs,
                                                                                 queryDiversityEta=eta)

            if self.args['smi_function'] == 'com':
                from submodlib_cpp import ConcaveOverModular
                obj = submodlib.ConcaveOverModularFunction(n=unlabeled_data_embedding.shape[0],
                                                           num_queries=query_embedding.shape[0],
                                                           query_sijs=query_sijs,
                                                           queryDiversityEta=eta,
                                                           mode=ConcaveOverModular.logarithmic)
            if self.args['smi_function'] == 'gcmi':
                obj = submodlib.GraphCutMutualInformationFunction(n=unlabeled_data_embedding.shape[0],
                                                                  num_queries=query_embedding.shape[0],
                                                                  query_sijs=query_sijs,
                                                                  metric=metric)
            if self.args['smi_function'] == 'logdetmi':
                lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 1
                obj = submodlib.LogDeterminantMutualInformationFunction(n=unlabeled_data_embedding.shape[0],
                                                                        num_queries=query_embedding.shape[0],
                                                                        data_sijs=data_sijs,
                                                                        query_sijs=query_sijs,
                                                                        query_query_sijs=query_query_sijs,
                                                                        magnificationEta=eta,
                                                                        lambdaVal=lambdaVal)

            greedyList = obj.maximize(budget=batch_size, optimizer=optimizer, stopIfZeroGain=stopIfZeroGain,
                                      stopIfNegativeGain=stopIfNegativeGain, verbose=verbose)
            self.top_idxs = [unlabeled[x[0]] for x in greedyList]
        self.idx = 0

    def sample(self, labeled_set):
        while self.top_idxs[self.idx] in labeled_set:
            self.idx += 1
        self.idx += 1
        return self.top_idxs[self.idx - 1]

    def __str__(self):
        return "SIMILAR"
