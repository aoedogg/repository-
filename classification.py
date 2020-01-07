import matplotlib.pyplot as plt
from matplotlib import style
import math
import time
from data_pre_processing import data_procss
import cvxpy as cvx
import numpy as np
import operator

# Decision Tree
class tree:
    def __init__(self, max_depth, min_size):
        self.max_depth = max_depth
        self.min_size = min_size

    def gini_index(self, groups, classes):
        # total number of data samples that we have
        tmp = ([len(gp) for gp in groups])
        n = float(sum(tmp))
        gini = 0.0
        for group in groups:
            lgroup = float(len(group)) # how many data points do we have in group
            # for each group we want to count number of elements classified in each class
            # NEED CASE for lgroup == 0
            score = 0.0
            if lgroup != 0:
                for cl in classes:
                    p = [row[-1] for row in group].count(cl)/lgroup
                    score += p*p
                gini += (1.0-score)*(lgroup/n)
        return gini

    # split list into two based on value evaluated at index
    def test_split(self, index, value, dataset):
        left = []
        right = []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def get_split(self, dataset_split):
        # given dataset, we must check each value on each attribute as split, eval cost of split and find
        # best split that we can make
        # for each row of dataset, compute gini for each element of that row
        cls = list(set([row[-1] for row in dataset_split])) # unique class values
        min_gini = 1000
        split = 1000
        split_index = 1000
        gp_min = []
        for row in dataset_split:
            ii = 0
            for element_s in row[0:-1]: # which value to split on
                # construct a group based on element
                index = ii
                left_s, right_s = self.test_split(index, element_s, dataset_split)
                gp = [left_s, right_s]
                gi = self.gini_index(gp, cls)
                print(element_s)
                print(gi)
                # compare split to see if lower than prev
                if gi < min_gini:
                    min_gini = gi
                    split = element_s
                    split_index = index
                    gp_min = gp
                ii = ii + 1
        #return split, split_index, gp ## change this to dictionary
        return {'index': split_index, 'value': split, 'groups': gp_min}

    def get_terminal(self, group):
        clasify = [row[-1] for row in group]
        count_max  = 0
        element_max = 0
        for ele in clasify:
            if clasify.count(ele) > count_max:
                count_max = clasify.count(ele)
                element_max = ele
        return element_max

    def split(self, node, depth):
        left, right = node['groups']
        del(node['groups'])
        if not left or not right: # if either left or right are empty
            node['left'] = node['right'] = self.get_terminal(left + right)
            return
        if depth >= self.max_depth:
            node['left'] = self.get_terminal(left)
            node['right'] = self.get_terminal(right)
            return
        if len(left) <= self.min_size:
            node['left'] = self.get_terminal(left)
        else:
            node['left'] = self.get_split(left) # define split and
            self.split(node['left'], depth+1)
        if len(right) <= self.min_size:
            node['right'] = self.get_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], depth+1)

        # no split - if either are empty
        # check if tree already larger than max
        # # process left child - if too few elements then terminal, else split
        # process right child - if too few elements then terminal, else split

    def build_tree(self, tt):
        root = self.get_split(tt) # get root node - will have left and right keys
        self.split(root, 1)
        return root

    def print_tree(self, node, depth=0):
        if isinstance(node, dict):
            # if node is a dictionary then it includes a split itself and sub. left/right splits
            print('Split: [X%d < %.3f]' % ((node['index'] + 1), node['value']))
            self.print_tree(node['left'], depth+1)
            self.print_tree(node['right'], depth + 1)
        else:
            print(node)

    def predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return predict(node['left'])
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return predict(node['right'])
            else:
                return node['right']

# K nearest neighbors
class kNN:
    def __init__(self, k):
        self.k = k
    # euclidean distance where we choose length of vector to compute distance for
    def euclidean_distance(self, x1, x2, l):
        dist = 0
        for ii in range(l):
            dist += pow(x1[ii] - x2[ii], 2)
        return math.sqrt(dist)
    # return k nearest neighbors of element in test Instance when compared to training set
    def getNeighbors(self, trainingSet, testInstance, l):
        dist = []
        for element in trainingSet:
            dist.append((element, self.euclidean_distance(element, testInstance, l)))
        # sort neighbors by distance
        dist.sort(key = lambda x: x[-1])
        # return k neighbors which are smallest distance
        dist = dist[0:self.k]
        return dist
    def getResponse(self, neighbors):
        vote = {} # dictionary of votes
        # iterate over k neighbors
        for ii in range(self.k):
            label = neighbors[ii][0][-1]
            if label in vote:
                vote[label] += 1
            else:
                vote[label] = 1
        # determine vote argument
        mx = -math.inf
        arg = []

        for key in vote:
            if vote[key] >= mx:
                mx = vote[key]
                arg = key

        return arg, vote


# State Vector Machine Classification - Dual formulation with kernal support
class SVM:
    pass

# State Vector Machine Classification - Primal formulation with no kernal support
class SVMp:
    def fit(self, train):
        # assume two classes and determine training split
        m = np.sum(train[:, -1] == 0)
        n = np.sum(train[:, -1] == 1)
        p = train.shape[1] - 1 # number of dimensions

        # split data
        x = train[train[:, -1] == 1,0:-1]
        y = train[train[:, -1] == 0,0:-1]

        # formulate optimization problem
        # these are our slack variables1
        u = cvx.Variable(m)
        v = cvx.Variable(n)
        a = cvx.Variable(p)
        b = cvx.Variable()
        constraints = []

        constraints += [a.T*x[ii, :] - b >= 1 - u[ii] for ii in range(m)]
        constraints += [a.T*y[ii, :] - b <= 1 + v[ii] for ii in range(n)]
        constraints += [u[i] >= 0  for i in range(m)]
        constraints += [v[i] >= 0  for i in range(n)]

        obj = cvx.Minimize(np.ones(m)*u + np.ones(n)*v)
        prob = cvx.Problem(obj, constraints)
        prob.solve(solver=cvx.CVXOPT, verbose=True)

        return a.value, b.value

# Naive Bayes Classification
class naive_bayes:
    # Adopted from: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

    def separatebyClass(self, data):
    # dataset = [[1,20,1], [2,21,0], [3,22,1]]
    # {0: [[2, 21, 0]], 1: [[1, 20, 1], [3, 22, 1]]}
    # separate based on last value
        n = len(data) # data input as a list
        separated = {} # dictionary of elements
        for ii in range(n): # iterate over all data elements in list
            element = data[ii][-1] # element we are trying to separate on
            if element in separated:
            # append dictionary
                separated[element].append(data[ii])
            else:
                # create element and append
                separated[element] = [data[ii]]
        # compute p(y) for each outcome
        p = {}
        for keys in separated:
            p[keys] = len(separated[keys])/n
        self.p = p

        return separated


    def mean(self, data):
        return sum(data) / float(len(data))

    def stdev(self, data):
        avg = self.mean(data)
        variance = sum(pow(x-avg, 2) for x in data)/float(len(data)-1)
        return math.sqrt(variance)

    def summarize(self, data):
        # dataset = [[1, 20, 0], [2, 21, 1], [3, 22, 0]]
        # Attribute summaries: [(2.0, 1.0), (21.0, 1.0)]
        # * unpacks tuple
        summaries = [(self.mean(attribute), self.stdev(attribute)) for attribute in zip(*data)]
        del summaries[-1]
        return summaries

    def summarizeByClass(self, data):
        # dataset = [[1,20,1], [2,21,0], [3,22,1], [4,22,0]]
        # 0: [(3.0, 1.4142135623730951), (21.5, 0.7071067811865476)], # 1: [(2.0, 1.4142135623730951), (21.0, 1.4142135623730951)]}
        seperated = self.separatebyClass(data)
        summaries = {}
        for key in seperated:
            tmp = self.summarize(seperated[key])
            summaries[key] = tmp
        return summaries

    def calculateProbability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def calculateClassProbabilities(self, summaries, inputVector):
        # summaries = {0:[(1, 0.5)], 1:[(20, 5.0)]}
        probabilities = {} # dictionary of probability
        n = len(inputVector)
        for key in summaries:
            probabilities[key] = 1
            for jj in range(len(summaries[key])): # for each input feature
                    x = inputVector[jj]
                    mean, std  = summaries[key][jj]
                    probabilities[key] *= self.calculateProbability(x, mean, std)
        return probabilities

    def predict(self, summaries, inputVector):
        nl = len(inputVector)
        out = []
        for jj in range(nl):
            inputfeature = inputVector[jj]
            probabilities = self.calculateClassProbabilities(summaries, inputfeature)
            bestLabel, bestProb = None, -1
            for key in probabilities:
                pxy = probabilities[key]*self.p[key]
                if bestLabel is None or pxy > bestProb:
                    bestProb = pxy
                    bestLabel = key
            out.append(bestLabel)
        return out

class K_means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def euclidean_distance(self, x1, x2):
        d = np.sqrt(np.dot(x1, x1) - 2 * np.dot(x1, x2) + np.dot(x2, x2))
        return d

    def predict(self, data):
        # find mean location which is shortest distance to each data point
        x = data
        # number of data points that we have
        n = x.shape[0]
        mindist = np.zeros([n])
        dist = np.zeros([n, self.k])
        for ii in range(n):
            for jj in range(self.k):
                dist[ii, jj] = self.euclidean_distance(data[ii, :], self.means[jj, :])
            # compute min distance to centroid
        mindist[:] = np.argmin(dist, axis=1)
        return mindist

    def fit(self, data):
        # assign each of our data point to a particular which is
        # the closest by euclidean distance
        x = data
        # number of data points that we have
        n = x.shape[0]
        # length of vectors we clustering
        p = x.shape[1]

        # assign initial means
        means = np.zeros([self.k, p])
        randnum = np.random.randint(low=0, high=n - 1, size=self.k)
        for ii in range(self.k):
            means[ii, :] = x[randnum[ii], :]

        kk = 1
        tol_e = np.inf


        dist = np.zeros([n, self.k])
        mindist = np.zeros([n])
        while kk <= self.max_iter:

        # assignment - assign each observation to the cluster whose mean has least squred eud distance
            for ii in range(n):
                for jj in range(self.k):
                    dist[ii, jj] = self.euclidean_distance(data[ii, :], means[jj, :])
        # compute min distance to centroid
            mindist[:] = np.argmin(dist, axis=1)
        # update mean
            for ii in range(self.k):
                dset = x[mindist == ii]
                # update mean
                means[ii, :] = np.sum(dset, axis=0)/dset.shape[0]

                self.means = means
            kk = kk + 1

dt = data_procss()
filename = 'diabetes.csv'
dataset = dt.loadCsv(filename)
nb = naive_bayes()
# summarize data by class
sm = nb.summarizeByClass(dataset)
test = list()
for element in dataset:
    test.append(element[0:-1])

# predict
pred = nb.predict(sm, test)

# compute accuracy
print(dt.classification_accuracy(dataset, pred))

# SVM


# Define the two sets
d = 2   # Dimension of problem. We'll leave at 2 for now.
m = 100 # Number of points in each class
n = 100

x_center = [1,1]  # E.g. [1,1]
y_center = [3,1]  # E.g. [2,2]

# Set a seed which will generate feasibly separable sets
#  Note: these may only be separable with the default tutorial settings
np.random.seed(8)

# Define random orientations for the two clusters
orientation_x = np.random.rand(2,2)
orientation_y = np.random.rand(2,2)

# Generate unit-normal elements, but clip outliers.
rx = np.clip(np.random.randn(m,d),-2,2)
ry = np.clip(np.random.randn(n,d),-2,2)
x = x_center + np.dot(rx,orientation_x)
y = y_center + np.dot(ry,orientation_y)



# Check out our clusters!
plt.scatter(x[:,0],x[:,1],color='blue')
plt.scatter(y[:,0],y[:,1],color='red')

# need to combine training data with labels
train = np.concatenate((x, y),axis=0)
lbl = np.concatenate((np.ones([n, 1]), np.zeros([m, 1])),axis=0)
train = np.concatenate((train, lbl), axis=1)

# train SVM
vm = SVMp()
a,b = vm.fit(train)
x1 = np.linspace(start=-3,stop=5,num=100)
x2 = [(1+b - a[0]*x1[ii])/a[1] for ii in range(100)]
# plot line
plt.plot(x1,x2)
#plt.show()

#Knn
kn = kNN(k = 3)
trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b'], [4.5, 4.5, 4.5, 'b'],[10,10,10,'a']]
testInstance = [5, 5, 5]
g = kn.getNeighbors(trainSet, testInstance, 3)
#print(g)
h, vote = kn.getResponse(g)

# Tree

# Determine gini index for different split configurations
#groups = [[[1, 1], [1, 0]], [[1, 1], [1, 0]]]
#classes = [0, 1]
#gi = tr.gini_index(groups, classes)
#print(gi)
# Want to determine one level split for this dataset
dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]]
# OUTPUT: Split: [X1 < 6.642]
#node = tr.get_split(dataset)
#        return {'index': split_index, 'value': element, 'groups': gp}
#print('Split: [X%d < %.3f]' % ((node['index']+1), node['value']))
#left, right = node['groups']
tr = tree(max_depth=1, min_size=1)
trained_tree = tr.build_tree(dataset)
tr.print_tree(trained_tree)
tr = tree(max_depth=2, min_size=1)
trained_tree = tr.build_tree(dataset)
tr.print_tree(trained_tree)