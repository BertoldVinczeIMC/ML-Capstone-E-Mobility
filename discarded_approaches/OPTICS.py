from __future__ import annotations

# Optics algorithm that will find the number of clusters and cluster them
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def optics_prep(data):
    # main function that will run the optics algorithm
    # data is the data that will be clustered

    clust = OPTICS(min_samples=2, xi=0.05, min_cluster_size=0.05)
    clust.fit(data)
    # create a numpy array from the labels
    labels = clust.labels_

    # num of clusters
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    return labels, n_clusters_
