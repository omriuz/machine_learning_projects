import matplotlib.pyplot as plt
import numpy as np
import random
from singlelinkage import singlelinkage


def main():
    data = np.load('mnist_all.npz')
    # create a random sample of 1000 samples, 100 from each digit
    X = np.concatenate(( data['train1'][:100],data['train0'][:100], data['train2'][:100], data['train3'][:100], data['train4'][:100],
                        data['train5'][:100], data['train6'][:100], data['train7'][:100], data['train8'][:100], data['train9'][:100]))
    # create the labels vector
    y = np.concatenate((np.ones((100, 1)), np.zeros((100, 1)), 2*np.ones((100, 1)), 3*np.ones((100, 1)), 4*np.ones((100, 1)),
                        5*np.ones((100, 1)), 6*np.ones((100, 1)), 7*np.ones((100, 1)), 8*np.ones((100, 1)), 9*np.ones((100, 1))))
    m, d = X.shape
    for k in [10,6]:
        C = singlelinkage(X, k=k)
        # calculate the number of samples in each cluster
        clusters = np.unique(C)
        num_samples = [np.sum(C == c) for c in clusters]
        print(f"Number of samples in each cluster: {num_samples}")
        # calculate the most common label in each cluster
        labels = np.unique(y)
        most_common_label = [labels[np.argmax(np.bincount(y[C == c].astype(int)))] for c in clusters]
        print(f"Most common label in each cluster: {most_common_label}")
        # calculate the accuracy of the clusteriing with percentage of correctly classified samples in each cluster
        percentages = [np.max(np.bincount(y[C == c].astype(int)))/np.sum(C == c)*100 for c in clusters]
        print(f"Percentage of points with most common label: {percentages}")
        # plot this information in a table
        fig, ax = plt.subplots()
        ax.axis('tight')
        ax.axis('off')
        cell_text = [[num_samples[i], percentages[i], most_common_label[i]] for i in range(len(clusters))]
        the_table = ax.table(cellText=cell_text, colLabels=['Number of samples', 'Percentage', 'Most common label'], loc='center')

        plt.show()
        sum_of_errors = sum([(1 - percentage/100) * num_samples for percentage, num_samples in zip(percentages, num_samples)])/m
        print(f"Sum of errors: {sum_of_errors}")




if __name__ == '__main__':
    main()