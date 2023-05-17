import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial import KDTree
from time import time

n = 5000
n_set = 1000
d = 4096

# data
X = np.random.randn(n, d)
X_set = np.random.randn(n_set, d)

# pairwise min_dist
start = time()
dist_ctr = pairwise_distances(X, X_set)
min_dist = np.amin(dist_ctr, axis=1)
print('Pairwise: ', time() - start)

start = time()
tree = KDTree(X_set)
min_dist_2, _ = tree.query(X, k=1)
print('KDTree: ', time() - start)

print('Max error: ', np.abs(min_dist - min_dist_2).max())