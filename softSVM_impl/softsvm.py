import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt
from tqdm import tqdm


def softsvm(l, trainX: np.array, trainy: np.array):
    """
    :param l: the parameter lambda of the soft SVM algorithm
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """
    m = trainX.shape[0]
    d = trainX.shape[1]

    # H is a matrix of size (m + d, m + d) where the first d rows and columns are 2*lambda and the rest is 0
    H = spmatrix(2 * l, range(d), range(d), (m + d, m + d))
    # u is a vector of size (m + d, 1) where the first d rows are 0 and the rest is 1 / m
    u = matrix(np.concatenate((np.zeros(d), np.ones(m) / m)))
    # A is a matrix of size (2m, m + d), divided to 4 blocks:
    top_left = matrix(np.zeros((m, d)))
    top_right = matrix(np.eye(m))
    bottom_left = matrix(np.array([trainy[i] * trainX[i] for i in range(m)]))
    bottom_right = matrix(np.eye(m))

    A = sparse([[top_left, bottom_left], [top_right, bottom_right]])

    # v is a vector of size (2m, 1) where the first m rows are 0 and the rest is 1
    v = matrix(np.concatenate((np.zeros(m), np.ones(m))))
    solvers.options['show_progress'] = False
    # solve the quadratic program
    sol = solvers.qp(H, u, -A, -v)

    # return the first d rows of the solution
    return np.array(sol['x'][:d])


def simple_test():
    data = np.load('mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")



def main():
    simple_test()


if __name__ == '__main__':
    main()
