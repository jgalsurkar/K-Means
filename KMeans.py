import numpy as np
from collections import Counter

class KMeans(object):
    """K-Means Clustering Model
    
    Parameters
    ----------        
    data : {array-like, sparse matrix},
    shape = [n_samples, n_features]
        data

    num_clusters : integer
        Number of clusters to cluster the data in
    """
    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters
        self.clusters = None
        self.objectives = None
        
    def closest_centroid(self, data):
        """Get the minimim distance from the data to each centroid

        Parameters
        ----------
        data : array-like, shape = [n_features]
            Data point
        
        Returns
        ----------
        min_centroid : Integer
            Closest centroid to the data point
        """
        distances = [np.square(np.linalg.norm(data - self.centroids[k])) for k in range(self.num_clusters)]
        min_centroid = np.argmin(np.array(distances))
        return min_centroid
        
    def assign_clusters(self):
        """Assigns a cluster to every data point
        
        Returns
        ----------
        clusters : array-like, shape = [n_samples]
            Assigned cluster for ever data point
        """
        clusters = np.apply_along_axis(self.closest_centroid, 1, self.data)
        return clusters

    def update_centroids(self, clusters):
        """ Update the vaue of each centroid given the assigned clusters

        Parameters
        ----------
        clusters : array-like, shape = [n_samples]
            Assigned cluster for ever data point
        """
        self.centroids = np.array([self.data[clusters == k].mean(axis = 0) for k in range(self.num_clusters)])

    def calculate_objective(self, clusters):
        """Calculate the k-means objective function value given the clusters

        Parameters
        ----------
        clusters : array-like, shape = [n_samples]
            Assigned cluster for ever data point
        
        Returns
        ----------
        obj : Float
            Value of objective function given the clusters
        """
        obj = sum([np.square(np.linalg.norm(self.data[clusters == k] - self.centroids[k])) for k in range(self.num_clusters)])
        return obj
    
    def find_clusters(self, iterations):
        """ Find the value of the centroid and assign clusters to ever data point

        Parameters
        ----------
        iterations : Integer
            Number of iterations of the k-means algorithm
        """
        self.objectives = []
        self.centroids = np.random.multivariate_normal(np.random.rand(1, 2)[0], [[1, 0], [0, 1]], self.num_clusters)
        for i in range(iterations):
            clusters = self.assign_clusters()
            self.update_centroids(clusters)
            self.objectives.append(self.calculate_objective(clusters))
        self.clusters = clusters
    
    def get_clusters(self):
        """ Get clusters for all data points

        Returns
        ----------
        clusters : array-like, shape = [n_samples]
            Assigned cluster for ever data point
        """
        return self.clusters
    
    def get_objectives(self):
        """ Get clusters for all data points

        Returns
        ----------
        objectives : List[Float]
            Objective function value at every iteration
        """
        return self.objectives