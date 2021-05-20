import numpy as np
import matplotlib.pyplot as plt
import math

"""
a1.py

This program implements least-squares linear classifier and  k-nearest neighbor classifier
and plots 1) the training samples 2) the decision boundary separating class 0 and 1, 
and 3) the classification regions.

@author: Anushree Das (ad1707)
"""


def least_squares(X,y):
    """
    Implements the least-squares linear classifier

    :param X: Input Features
    :param y: Output Class
    :return:
    """
    print(' Least-Squares Linear Classifier '.center(50, '#'))
    # add extra column of ones to X for calculating dot product with weights with bias bit
    X_new = np.hstack((np.ones((len(X), 1)), X))
    # calculate weights
    beta = (np.linalg.inv(X_new.T.dot(X_new)).dot(X_new.T).dot(y))

    # plot decision boundary and classification region
    plot_training_samples(X, y)
    plot_classification(X,y,'ls',beta=beta)
    y_pred = least_squares_predict(X_new,beta)
    confusion_matrix(y,y_pred)


def k_nearest_neighbors(X,y,n):
    """
    Implements the k-nearest neighbor classifier

    :param X: Input Features
    :param y: Output Class
    :param n: Number of nearest neighbors
    :return:
    """
    print('\n',(' '+str(n)+'- Nearest Neighbor Classifier ').center(50, '#'))
    # plot decision boundary and classification region
    plot_training_samples(X, y)
    plot_classification(X,y,str(n)+'nn',n=n)
    y_pred = nearest_neighbors_predict(X,y,X, n)
    confusion_matrix(y, y_pred)


def plot_training_samples(X, y):
    """
    Plots the training samples

    :param X: Input Features
    :param y: Output Class
    :return:
    """
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')

    # assign color for each input according to its output class
    colors = get_colors(y)
    # plot features
    plt.scatter(X[:, 0], X[:, 1],marker='o', s=20, facecolors='none',edgecolors=colors)


def plot_classification(X,y,name,beta=None,n=0):
    """
    Plots the decision boundary and classification region

    :param X: Input Features
    :param y: Output Class
    :param name: plot name
    :param beta: weights for least square classifier
    :param n: Number of nearest neighbors for n-nearest neighbors
    :return:
    """
    # find min and max values of both features
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))

    if beta is None:
        # k-nearest neighbors classifier
        X_all = np.c_[x_vals.ravel(), y_vals.ravel()]
        # Predict class for all combinations of values of both features
        predictions = nearest_neighbors_predict(X, y, X_all, n)
        z = predictions.reshape(x_vals.shape)
        # draw decision boundary
        plt.contour(x_vals, y_vals, z,linewidths=0.5,levels=[0.9,1], colors=['black'])
    else:
        # least-squares linear classifier
        # different x and y values to get smooth line
        x_vals_new, y_vals_new = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        X_all = np.c_[np.ones((len(x_vals_new.ravel()), 1)),x_vals_new.ravel(), y_vals_new.ravel()]
        # Predict class for all combinations of values of both features
        predictions = least_squares_predict(X_all, beta)
        z = predictions.reshape(x_vals_new.shape)
        # draw decision boundary
        plt.contour(x_vals_new, y_vals_new, z, linewidths=0.5,levels = [0.9,1], colors=['black'])

        X_all = np.c_[np.ones((len(x_vals.ravel()), 1)), x_vals.ravel(), y_vals.ravel()]
        # Predict class for all combinations of values of both features
        predictions = least_squares_predict(X_all,beta)

    # assign color for each input according to its output class
    colors = get_colors(predictions)
    # plot classification region
    plt.scatter(x_vals, y_vals, s=0.1, color=colors)

    # plt.savefig(name+'_decision_boundary.png')
    plt.show()


def get_colors(classlabels):
    """
    Returns array of colors based on output labels

    :param classlabels: Output labels
    :return: array of colors
    """
    # assign color for each input according to its output class
    colors = []
    for c in classlabels:
        if c == 0:
            colors.append('skyblue')
        else:
            if c == 1:
                colors.append('orange')
    return colors


def least_squares_predict(X,beta):
    """
    Predict class using least square classifier weights
    :param X: Input Features
    :param beta: Weights
    :return: Output labels
    """
    y_pred = threshold(X.dot(beta))
    return y_pred


def threshold(y_pred):
    """
    Threshold the class probabilities to categorize into class labels 0 or 1

    :param y_pred: array of class probabilities
    :return: array of class labels
    """
    y_pred_thresh = np.zeros(y_pred.shape)
    for i in range(len(y_pred)):
        if y_pred[i] > 0.5:
            y_pred_thresh[i] = 1
        else:
            y_pred_thresh[i] = 0
    return y_pred_thresh


def nearest_neighbors_predict(X_train,y_train,X_test,n):
    """
    Predict class using n-nearest neighbors classifier

    :param X_train: Training sample input features
    :param y_train: Training sample output labels
    :param X_test: Test sample input features
    :param n: Number of nearest neighbors for n-nearest neighbors
    :return: Test sample output labels
    """
    y_pred = []
    # for every features vector in test sample
    for test_sample in X_test:
        dist = []
        # calculate its distance from every features vector in training sample
        for train_sample in X_train:
            dist.append(euclidean_dist(test_sample,train_sample))
        # find n nearest neighbors
        asc_order = np.argsort(dist)
        nearest_labels = []
        for i in range(n):
            nearest_labels.append(y_train[asc_order[i]])
        # assign the most common class label from the n nearest neighbors
        y_pred.append(max(nearest_labels, key = nearest_labels.count))

    return np.array(y_pred)


def euclidean_dist(x,y):
    """
    Calculates euclidean distance between two points x and y
    :param x: first point
    :param y: second point
    :return: distance between two points
    """
    return math.sqrt(((x[0]-y[0])**2)+((x[1]-y[1])**2))


def confusion_matrix(y,y_pred):
    correct_0 = 0
    correct_1 = 0
    wrong_0 = 0
    wrong_1 = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y[i]:
            if y[i] == 0:
                correct_0 += 1
            else:
                correct_1 += 1
        else:
            if y[i] == 0:
                wrong_0 += 1
            else:
                wrong_1 += 1

    print("{:<15} {:^20}".format(' ', 'Ground Truth'))
    print("{:>15} {:^10} {:^10}".format('Predictions', 'Blue(0)', 'Orange(1)'))
    print('_'*40)
    print("{:>15} {:^10} {:^10}".format('Blue(0)',correct_0,wrong_1))
    print("{:>15} {:^10} {:^10}".format('Orange(1)',wrong_0,correct_1))
    print('_' * 40)

def main():
    # load data
    data = np.load("data.npy")
    # array of features
    X = data[:,:-1]
    # array of output class for corresponding feature set
    y = data[:,-1]

    # least-squares linear classifier
    least_squares(X,y)
    # 1-nn (nearest neighbor) classifier
    k_nearest_neighbors(X, y, 1)
    # 15-nn (nearest neighbor) classifier
    k_nearest_neighbors(X, y, 15)


if __name__ == '__main__':
    main()
