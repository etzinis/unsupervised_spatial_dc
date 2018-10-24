"""!
@brief This utility serves as a level of abstraction in order to
construct audio mixtures


@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana Champaign
"""

from pprint import pprint
from sklearn.cluster import KMeans
import numpy as np


class RobustKmeans(object):
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

    def fit(self, x, cut_outlier_in_norm=2.):
        """!
        robust clustering for the input x

        :param x: nd array with shape: (n_samples, n_features)

        :return cluster_labels: 1d array with the corresponding
        labels from 0 to self.N_true - 1
        """

        if cut_outlier_in_norm is not None:
            robust_points = x[np.where(np.linalg.norm(x, axis=1) <=
                              cut_outlier_in_norm), :][0]

            fitted_centers = self.kmeans_obj.fit(robust_points)
            clustered = self.kmeans_obj.predict(x)
        else:
            fitted_centers = self.kmeans_obj.fit(x)
            clustered = fitted_centers.labels_

        cluster_coordinates = fitted_centers.cluster_centers_

        priors = np.bincount(clustered)
        cl_indexes = np.argsort(priors)
        true_clusters = cl_indexes[self.N_used - self.N_true:]

        fitted_centers.cluster_centers_ = cluster_coordinates[
                                          true_clusters]

        # make the new prediction with the new clusters
        robust_estimation = fitted_centers.predict(x)

        return robust_estimation

    def fit_predict(self, x, cut_outlier_in_norm=2.):
        """!
        robust clustering for the input x

        :param x: nd array with shape: (n_samples, n_features)

        :return cluster_labels: 1d array with the corresponding
        labels from 0 to self.N_true - 1
        """
        return self.fit(x, cut_outlier_in_norm=cut_outlier_in_norm)


def example_of_usage():
    """!
    How the class of Audio mixtures should be called"""

    from sklearn.datasets import load_iris
    data = load_iris()
    x = data.data
    y = data.target
    x /= np.linalg.norm(x)

    robust_clusterer = RobustKmeans(n_true_clusters=3,
                                    n_used_clusters=3)
    pred = robust_clusterer.fit(x)
    print("Using 3 True Clusters and 3 for Prediction: {}".format(pred))

    robust_clusterer = RobustKmeans(n_true_clusters=3,
                                    n_used_clusters=5)
    pred = robust_clusterer.fit(x)
    print("Using 3 True Clusters and 5 for Prediction: {}".format(pred))

if __name__ == "__main__":
    example_of_usage()