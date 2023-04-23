import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import teg_CCC
import importlib
importlib.reload(teg_CCC)

plots0 = False
n_Obs = 1000
n_Features = 3
noise0 = 0.5
n_iters = 20
results = []
for n_clusters_true in range(2, 6):
    print(n_clusters_true)
    new_row = []
    for i_iter in range(n_iters):
        X, labels_true = make_blobs(n_samples=n_Obs, n_features=n_Features, centers=n_clusters_true, random_state=0, cluster_std=noise0)
        O = teg_CCC.get_best_k_CCC(X)
        best_n = O['best_n']

        print(n_clusters_true, best_n)
        new_row.append(best_n)

        if plots0:
            km = KMeans(n_clusters=n_clusters_true, random_state=0, n_init="auto").fit(X)
            fig, ax = plt.subplots(2, 1)
            ax[0].scatter(X[:, 0], X[:, 1], s=10, c=km.labels_)
            ax[1].plot(O['n_vec'], O['scores_per_n'])
            plt.show()
    results.append([n_clusters_true, new_row])
