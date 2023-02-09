import numpy as np



def singlelinkage(X, k):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    m = X.shape[0]
    dist = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            dist[i, j] = np.linalg.norm(X[i] - X[j])
    np.fill_diagonal(dist, np.inf)
    clusters = np.array(range(m))
    for _ in range(X.shape[0] - k):
        idx_1, idx_2 = np.unravel_index(np.argmin(dist), (m, m))
        dist[idx_1, :] = np.inf
        dist[:, idx_1] = np.inf
        clusters[clusters == idx_1] = idx_2
        for i in range(m):
            if i != idx_1 and i != idx_2:
                temp = min(dist[idx_1][i], dist[idx_2][i])
                dist[idx_2][i] = temp
                dist[i][idx_2] = temp
    return clusters.reshape(m, 1)


def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    # create a random sample of 300 samples, 30 from each digit
    X = np.concatenate(
        (data['train0'][:30], data['train1'][:30], data['train2'][:30], data['train3'][:30], data['train4'][:30],
         data['train5'][:30], data['train6'][:30], data['train7'][:30], data['train8'][:30], data['train9'][:30]))
    m, d = X.shape

    # run single-linkage
    c = singlelinkage(X, k=10)

    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2
