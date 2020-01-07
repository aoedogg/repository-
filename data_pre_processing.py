from random import random
import csv

class data_procss:

    def loadCsv(self, filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
        dataset = list(lines)
        dataset = dataset[1:-1]
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]

        return dataset

    # split dataset into list of training data and testing data
    # split is how much should be training set
    def loadDataset(self, filename, split, trainingSet=[], testSet=[]):
        # load dataset
        dataset = self.loadCsv(filename)
        n = len(dataset)

        for ii in range(n):
            # generate random number
            rand = random()
            if rand > split:
                # testing
                testSet.append(dataset[ii])
            else:
                # train
                trainingSet.append(dataset[ii])

    # compute the accuracy of a model where last column of data is the actual output
    def classification_accuracy(self, data, output):
        labels = []
        n = len(output)
        for element in data:
            labels.append(element[-1])
        cmp = [x == y for (x, y) in zip(output, labels)]
        s = 0

        for element in cmp:
            if element is True:
                s = s + 1
        return s/n
