import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


def softsvmpoly(l: float, k: int, trainX: np.array, trainy: np.array):
    """
    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    d = len(trainX[0])
    m = len(trainX)
    G = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            G[i][j] = (1 + np.inner(trainX[i], trainX[j])) ** k
    A = matrix(np.block([[np.zeros((m, m)), np.eye(m)], [np.diag(trainy) @ G, np.eye(m)]]))
    H = np.block([[2 * l * G, np.zeros((m, m))], [np.zeros((m, m)), np.zeros((m, m))]])
    epsilon_matrix = np.eye(2 * m) * 1e-5
    H = matrix(H + epsilon_matrix)
    u = matrix(np.concatenate((np.zeros(m), np.ones(m) * 1 / m)))
    v = matrix(np.concatenate((np.zeros(m), np.ones(m))))
    solvers.options['show_progress'] = False
    sol = solvers.qp(H, u, -A, -v)
    alpha = np.array(sol["x"][:m])
    return alpha


def simple_test():
    data = np.load('mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvmpoly algorithm
    w = softsvmpoly(10, 5, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


def predict(alpha: np.array, trainX: np.array, k):
    def predictor(x):
        ans = 0
        for i in range(trainX.shape[0]):
            ans += alpha[i] * (1 + np.inner(trainX[i], x)) ** k
        return np.sign(ans)
    return predictor


if __name__ == '__main__':
    simple_test()
