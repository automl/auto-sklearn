__authors__ = ["Matthias Feurer"]

import numpy as np
import scipy.stats
import sklearn.cluster
import sklearn.manifold
import sklearn.preprocessing
import sklearn.utils


class GMeans(object):
    def __init__(self, minimum_samples_per_cluster=2, n_init=10, significance=4,
                 restarts=10, random_state=None, ):
        self.minimum_samples_per_cluster = minimum_samples_per_cluster
        self.n_init = n_init
        self.significance = significance
        self.restarts = restarts
        self.random_state = sklearn.utils.check_random_state(random_state)

    def fit(self, X):
        self.inertia_ = np.inf

        for i in range(self.restarts):
            KMeans = sklearn.cluster.KMeans(n_clusters=1, n_init=1,
                                            random_state=self.random_state)
            KMeans.fit(X)

            while True:
                # Splitting part
                change = False
                cluster_centers = []

                for i, cluster_center in enumerate(KMeans.cluster_centers_):
                    indices = KMeans.labels_ == i
                    X_ = X[indices]

                    for i in range(10):
                        KMeans_ = sklearn.cluster.KMeans(n_clusters=2,
                                                         n_init=self.n_init,
                                                         random_state=self.random_state)
                        predictions = KMeans_.fit_predict(X_)
                        bins = np.bincount(predictions)
                        minimum = np.min(bins)
                        if minimum >= self.minimum_samples_per_cluster:
                            break

                    if minimum <= self.minimum_samples_per_cluster:
                        cluster_centers.append(cluster_center)
                    else:
                        # Test fit:
                        centroid0, centroid1 = KMeans_.cluster_centers_
                        v = centroid1 - centroid0
                        X__prime = np.inner(v, X_) / np.linalg.norm(v, ord=2)
                        mean = np.mean(X__prime)
                        std = np.std(X__prime)
                        X__prime = (X__prime - mean) / std
                        # A2 is A^2_* from [Hamerly 2006], equation (2)
                        A2, critical, sig = scipy.stats.anderson(X__prime)

                        # Reject the split
                        if A2 < critical[self.significance]:
                            cluster_centers.append(cluster_center)
                        # Accept the split
                        else:
                            change = True
                            cluster_centers.extend(KMeans_.cluster_centers_)

                if change is False:
                    break

                # Refinement
                KMeans = sklearn.cluster.KMeans(n_clusters=len(cluster_centers), n_init=1,
                                                init=np.array(cluster_centers),
                                                random_state=self.random_state)
                KMeans.fit(X)

            if KMeans.inertia_ < self.inertia_:
                self.KMeans = KMeans
                self.inertia_ = self.KMeans.inertia_

        self.cluster_centers_ = self.KMeans.cluster_centers_
        self.labels_ = self.KMeans.labels_
        self.inertia_ = self.KMeans.inertia_

    def fit_predict(self, X):
        self.fit(X)
        predictions = self.KMeans.predict(X)
        return predictions
