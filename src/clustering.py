from __future__ import annotations

import numpy as np
from aeon.clustering.k_means import TimeSeriesKMeans
from matplotlib import pyplot as plt
import polars as pl

from data import datasets, WallboxesColumns


def cluster(df: pl.DataFrame, column: str, k: int = 2):
    """
    Clusters the given dataframe using the k-means algorithm

    :param df: The dataframe to cluster
    :param column: The column to cluster
    :param k: The number of clusters
    :return: The dataframe with an additional column containing the cluster
    """
    minutes_in_a_day = 1440
    days = 417

    filled = df.upsample(
        time_column="Timestamp",
        every="1m",
    ).select(column)

    days_df = []

    for i in range(days):
        computed = (
            filled.slice(i * minutes_in_a_day, minutes_in_a_day)
            .fill_nan(1100.0)
            .to_numpy()
        )

        if np.count_nonzero(np.isnan(computed)) == 0:
            days_df.append(computed)

    days_df_np = np.array(days_df)

    normalized = (days_df_np - days_df_np.mean(axis=0)) / days_df_np.std(axis=0)

    kmeans = TimeSeriesKMeans(
        n_clusters=k, metric="dtw", max_iter=5, random_state=69420
    )

    print(normalized.shape)

    clusters = kmeans.fit_predict(normalized)

    # plots
    first_cluster = np.where(clusters == 0)[0]

    second_cluster = np.where(clusters == 1)[0]

    for i in first_cluster:
        plt.plot(normalized[i], color="red", alpha=0.2)
        plt.xlabel("Time (minutes)")
        plt.ylabel("Power (kW)")

    plt.title(f"Clustering of {column} with k={k}. Cluster #1")

    # plt.show()

    plt.savefig(f"plots/clustering/{column}_{k}_cluster1.png")

    # clear plot
    plt.clf()

    for i in second_cluster:
        plt.plot(normalized[i], color="blue", alpha=0.2)
        plt.xlabel("Time (minutes)")
        plt.ylabel("Power (kW)")

    plt.title(f"Clustering of {column} with k={k}. Cluster #2")

    plt.savefig(f"plots/clustering/{column}_{k}_cluster2.png")

    # plt.show()

    plt.clf()


if __name__ == "__main__":
    cluster(datasets.wallboxes, WallboxesColumns.DELTA)
    cluster(datasets.wallboxes, WallboxesColumns.RAPTION)
    cluster(datasets.wallboxes, WallboxesColumns.KEBA_ONE)
    cluster(datasets.wallboxes, WallboxesColumns.KEBA_TWO)
    cluster(datasets.wallboxes, WallboxesColumns.KEBA_THREE)
    cluster(datasets.wallboxes, WallboxesColumns.LADEBOX_ONE)
    cluster(datasets.wallboxes, WallboxesColumns.LADEBOX_TWO)
    cluster(datasets.wallboxes, WallboxesColumns.LADEBOX_THREE)
