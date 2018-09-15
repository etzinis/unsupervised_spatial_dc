"""!
@brief This utility serves as a level of abstraction in order to
construct audio mixtures


@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana Champaign
"""

from pprint import pprint
from sklearn.cluster import KMeans
import numpy as np


class RobustClustering(object):
    def __init__(self,
                 n_true_clusters=2,
                 n_used_clusters=4):
        """!
        Sometimes K-means creates clusters around outlier groups which
        should not be the case. For this reason we run K-means with
        n_used_clusters > n_true_clusters and then we assign at the most
        probable n_true_clusters the residual clusters

        :param n_true_clusters: the true number of clusters we wanna
        cluster the data at the end
        :param n_used_clusters: The amount of clusters that will be used
        in total for running kmeans and after that the residual would be
        assigned in the top most prior n_true_clusters
        """

        self.N_true = n_true_clusters
        self.N_used = n_used_clusters
        self.kmeans_obj = KMeans(n_clusters=self.N_used,
                                 random_state=7)

    def robust_kmeans(self, x):
        """!
        robust clustering for the input x

        :param x: nd array with shape: (n_samples, n_features)

        :return cluster_labels: 1d array with the corresponding
        labels from 0 to self.N_true - 1
        """

        clustered = self.kmeans_obj.fit(x).labels_
        priors = np.bincount(clustered)
        cl_indexes = np.argsort(priors)
        residual_clusters = cl_indexes[:-self.N_true]

        true_clusters = cl_indexes[self.N_used-self.N_true:]
        cluster_identity = dict([(i, k)
                                 for k, i in enumerate(true_clusters)])
        top_prior = cluster_identity[cl_indexes[-1]]

        cluster_changer = dict([(i, top_prior)
                                for i in residual_clusters])
        cluster_changer.update(cluster_identity)

        robust_clustered = [cluster_changer[cl_ind]
                            for cl_ind in clustered]

        return np.array(robust_clustered)


def example_of_usage():
    """!
    How the class of Audio mixtures should be called"""

    from sklearn.datasets import load_iris
    data = load_iris()
    x = data.data
    y = data.target

    robust_clusterer = RobustClustering(n_true_clusters=3,
                                        n_used_clusters=3)
    pred = robust_clusterer.robust_kmeans(x)
    print("Using 3 True Clusters and 3 for Prediction: {}".format(pred))

    robust_clusterer = RobustClustering(n_true_clusters=3,
                                       n_used_clusters=5)
    pred = robust_clusterer.robust_kmeans(x)
    print("Using 3 True Clusters and 3 for Prediction: {}".format(pred))

    robust_clusterer = RobustClustering(n_true_clusters=3,
                                       n_used_clusters=10)
    pred = robust_clusterer.robust_kmeans(x)
    print("Using 3 True Clusters and 3 for Prediction: {}".format(pred))

if __name__ == "__main__":
    example_of_usage()