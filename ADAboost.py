import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz

# CHANGE TO CLASS BASED STRUCTURE
# EVENTUALLY IMPLEMENT DECISION STUMP SEPARATE

#Toy Dataset
x1 = np.array([.1,.2,.4,.8, .8, .05,.08,.12,.33,.55,.66,.77,.88,.2,.3,.4,.5,.6,.25,.3,.5,.7,.6])
x2 = np.array([.2,.65,.7,.6, .3,.1,.4,.66,.77,.65,.68,.55,.44,.1,.3,.4,.3,.15,.15,.5,.55,.2,.4])
y = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
X = np.vstack((x1, x2)).T

n = x1.shape[0]

# plot without any classification boundary
plt.scatter(x1, x2, c=y)
plt.show()

# number of boosting cycles
m = 3
w = np.zeros([m +1, n])
# initial weights
w[0, :] = np.ones([n])*1/n
# list of classifier
clf_list = []
# alpha
alpha = np.zeros([1, m])
for ii in range(m):
    clf = tree.DecisionTreeClassifier(max_depth=1)
    clf = clf.fit(X, y, sample_weight=w[ii, :])
    # print score associated with stage
    #print(clf.score(X, y, sample_weight=w[ii, :]))
    #print(np.sum(clf.predict(X) == y)/n)
    # append list with new model
    clf_list.append(clf)
    # compute weighted error
    if_correct = clf.predict(X) != y
    sum_correct = np.dot(w[ii, :], if_correct)/np.sum(w[ii, :])
    # compute coefficient
    alpha[0, ii] = np.log((1-sum_correct)/sum_correct)
    #weight update
    w[ii+1,:] = np.multiply(w[ii, :], np.exp(alpha[0, ii]*if_correct))

# create boosted classifier
pred = np.zeros([1, n])
for ii in range(m):
    clf = clf_list[ii]
    pred[0, :] = pred[0, :] + alpha[0, ii]*clf.predict(X)

out = np.sign(pred[0, :])
print(np.sum(out == y) / n)

