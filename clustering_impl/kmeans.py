import numpy as np
import random


def kmeans(X, k, t):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :param t: the number of iterations to run
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    # init random centroids based on the sample
    m, d = X.shape
    centroids = np.zeros((k, d))
    for i in range(k):
        centroids[i] = X[random.randint(0, m - 1)]
    # run t iterations
    for i in range(t):
        # calculate the distance of each sample to each centroid
        dist = np.zeros((m, k))
        for j in range(k):
            dist[:, j] = np.linalg.norm(X - centroids[j], axis=1)
        # assign each sample to the closest centroid
        c = np.argmin(dist, axis=1)
        # update the centroids
        for j in range(k):
            centroids[j] = np.mean(X[c == j], axis=0)
    return c.reshape(m, 1)


def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    X = np.concatenate((data['train0'], data['train1']))
    m, d = X.shape

    # run K-means
    c = kmeans(X, k=10, t=10)

    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    simple_test()