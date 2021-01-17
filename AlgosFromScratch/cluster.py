import matplotlib.pyplot as plt
from matplotlib import style
import math
import time
from data_pre_processing import data_procss
import cvxpy as cvx
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Gaussian Mixture Model (GMM)
# Based on tutorial: http://www.oranlooney.com/post/ml-from-scratch-part-5-gmm/
class GMM:
    def __init__(self, k, max_iter=10):
        self.k = k
        self.max_iter = int(max_iter)

    def initialize(self, X):
        self.n, self.m = X.shape
        self.phi = np.full(shape=self.k, fill_value=1/self.k)
        self.weights = np.full(shape=(self.n, self.k),fill_value=1/self.k)

        random_row = np.random.randint(low=0, high=self.n, size=self.k)
        self.mu = [X[row_index, :] for row_index in random_row]
        self.sigma = [np.cov(X.T) for _ in range(self.k)]


    def e_step(self, X):
        # Given fixed class probabilities, class means and class SDs, compute new weights - prob that certain
        # value from training data belongs to certain class. This formula is derived from Bayes rule.
        self.weights = self.predict_prob(X) # X is n*m
        self.phi = self.weights.mean(axis=0)

    def predict_prob(self, X):
        # Given mean and cov for each class, compute pdf eval of data X
        # n --> number of data points
        # k --> number of classes we will classify into
        liklihood = np.zeros((self.n, self.k))
        for jj in range(self.k):
            dist = multivariate_normal(mean = self.mu[jj], cov = self.sigma[jj])
            liklihood[:, jj] = dist.pdf(X) # n * m matrix

        num = liklihood * self.phi # scale each column by probability of belonging to cluster
        den = num.sum(axis=1)[:, np.newaxis] # add additional dimension so that we can use / operator
        weights = num / den
        return weights

    def m_step(self, X):
        for jj in range(self.k):
            weight = self.weights[:, [jj]] # choose weights associated with each cluster
            t_weight = weight.sum()
            # update mu - this is a MLE based on assumption of fixed weights
            self.mu[jj] = (X*weight).sum(axis=0)/ t_weight
            self.sigma[jj] = np.cov(X.T, aweights=(weight / t_weight).flatten(), bias=True)
            # write covariance function from scratch

    def fit(self, X):
        self.initialize(X)
        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)

    def predict(self, X):
        weights = self.predict_prob(X)
        return np.argmax(weights, axis=1)

# Generate some data
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.60, random_state=0)
X = X[:, ::-1] # flip axes for better plotting

gmm = GMM(k=4, max_iter=100)
gmm.fit(X)
labels = gmm.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.show()