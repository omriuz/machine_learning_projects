import collections
import numpy as np
from scipy.spatial import distance
from heapq import heappush, heappop
import matplotlib.pyplot as plt
import random


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m alongside its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


def gensmallm_with_corruption(x_list: list, y_list: list, m: int):
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])
    to_corrupt = random.sample(range(len(y)), int(len(y) * 0.15))

    for i in to_corrupt:
        labels = [2, 3, 5, 6]
        labels.remove(y[i])
        y[i] = labels[random.randrange(3)]

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


class classifier:
    def __init__(self, k, x_train, y_train):
        self.k = k
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        y_pred = []
        for x in x_test:
            y_pred.append(self.predict_one(x))
        return y_pred

    def predict_one(self, x):
        dists = [(distance.euclidean(example, x), y) for example, y in zip(self.x_train, self.y_train)]
        kheap = []
        for dist, y in dists:
            if (len(kheap) < self.k):
                heappush(kheap, (-dist, y))
            elif dist < -kheap[0][0]:
                heappop(kheap)
                heappush(kheap, (-dist, y))
        return collections.Counter([y for _, y in kheap]).most_common(1)[0][0]


def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    return classifier(k, x_train, y_train)


def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    return np.transpose(np.array([classifier.predict(x_test)]))


def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")


def nearset_neighbor():
    data = np.load('mnist_all.npz')
    avg_errors = []
    max_errors = []
    min_errors = []
    (x_test, y_test) = gensmallm([data['test2'], data['test3'], data['test5'], data['test6']], [2, 3, 5, 6],
                                 len(data['test2']) + len(data['test3']) + len(data['test5']) + len(data['test6']))
    repeats = 10
    rng = range(1, 101, 5)
    k = 1
    for m in rng:
        sum_error = 0
        max_error = 0
        min_error = 1
        for _ in range(repeats):
            (x_train, y_train) = gensmallm([data['train2'], data['train3'], data['train5'], data['train6']],
                                           [2, 3, 5, 6], m)
            classifier = learnknn(k, x_train, y_train)
            y_pred = predictknn(classifier, x_test).transpose()[0]
            cur_error = np.mean(y_test != y_pred)
            max_error = max(max_error, cur_error)
            min_error = min(min_error, cur_error)
            sum_error += cur_error

        max_errors.append(max_error)
        min_errors.append(min_error)
        avg_errors.append(sum_error / repeats)

    plot_max = plt.bar(rng, max_errors, 2)
    plot_min = plt.bar(rng, min_errors, 2)
    plot_avg = plt.plot(rng, avg_errors, 'r')
    plt.legend((plot_avg[0], plot_max[0], plot_min[0]), ('Average error', 'Max error', 'Min error'))
    plt.xlabel('Sample Size')
    plt.ylabel('Test prediction error')
    plt.title('Question 2a')
    plt.show()


def k_nearest_neighbors():
    data = np.load('mnist_all.npz')
    avg_errors = []
    (x_test, y_test) = gensmallm([data['test2'], data['test3'], data['test5'], data['test6']], [2, 3, 5, 6],
                                 len(data['test2']) + len(data['test3']) + len(data['test5']) + len(data['test6']))
    repeats = 10
    rng = range(1, 12)
    for k in rng:
        sum_error = 0
        for i in range(repeats):
            (x_train, y_train) = gensmallm([data['train2'], data['train3'], data['train5'], data['train6']],
                                           [2, 3, 5, 6], 200)
            classifier = learnknn(k, x_train, y_train)
            y_pred = predictknn(classifier, x_test).transpose()[0]
            cur_error = np.mean(y_test != y_pred)
            sum_error += cur_error

        avg_errors.append(sum_error / repeats)

    plt.plot(rng, avg_errors)
    plt.xlabel('K')
    plt.ylabel('Test prediction error')
    plt.title('Question 2e')
    plt.show()


def k_nearest_neighbors_corrupt():
    data = np.load('mnist_all.npz')
    avg_errors = []
    (x_test, y_test) = gensmallm([data['test2'], data['test3'], data['test5'], data['test6']], [2, 3, 5, 6],
                                 len(data['test2']) + len(data['test3']) + len(data['test5']) + len(data['test6']))
    repeats = 10
    rng = range(1, 12)
    for k in rng:
        sum_error = 0
        for _ in range(repeats):
            (x_train, y_train) = gensmallm_with_corruption(
                [data['train2'], data['train3'], data['train5'], data['train6']],
                [2, 3, 5, 6], 100)
            classifier = learnknn(k, x_train, y_train)
            y_pred = predictknn(classifier, x_test).transpose()[0]
            cur_error = np.mean(y_test != y_pred)
            sum_error += cur_error

        avg_errors.append(sum_error / repeats)

    plt.plot(rng, avg_errors)
    plt.xlabel('K')
    plt.ylabel('Test prediction error')
    plt.title('Question 2f')
    plt.show()


if __name__ == '__main__':
    simple_test()
    nearset_neighbor()
    k_nearest_neighbors()
    k_nearest_neighbors_corrupt()
