# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 10:51:40 2018

@author: Manuel Seet

@description: Personal learning ex to build 
a perceptron from scratch based on the concepts learned


"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

###############Basics Functions ################################################

# define a sigmoidal function


def sigmoid(x):
    s = 1 / (1+np.exp(-x))
    return s


def sigmoid_der(x):  # derivative
    d = sigmoid(x) * (1 - sigmoid(x))
    return d


#################################################################################
####################### Perceptron #######################################
#################################################################################

# 1-layer perceptron: 3 inputs, 1 output

# create a perceptron class, so that we can instantiate
class Perceptron():
    # so I want to set the num of features and the learning rate
    def __init__(self, nFeatures, bias, lr):
        self.weights = np.random.rand(nFeatures, 1)  # set random weights
        self.bias = bias  # get some bias
        self.learnRate = lr  # set learning rate

    # multiply each input with respective weight, get a number, put it thru sigmoid to get output
    def predict(self, inputs):
        raw = np.dot(inputs, self.weights) + self.bias
        output = sigmoid(raw)
        return output

    def train(self, features, label, numEpochs):
        for i in range(numEpochs):
            # step 1 Forward Propagation
            pred = self.predict(features)  # get the prediction values
            # step 2 Cost calculation
            error = label-pred
            # step3 Back propagation
            dcost_dpred = error
            dpred_draw = sigmoid_der(pred)
            delta = dcost_dpred * dpred_draw  # apply chain rule here
            # calculate the change value, moderated by learning rate
            change = self.learnRate * np.dot(features.T, delta)
            self.weights += change  # adjust the weights
        return error  # return the final error


#### TRAINING THE PERCEPTRON ##################
trainRounds = 250  # 250
trainRoundList = list(range(1, trainRounds+1))
outputList = []
errorList = []

# make a dummy dataset to train the perceptron
features = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])

labels = np.array([[0, 1, 1, 0]]).T
nFeatures = features.shape[1]
lr = 0.05  # set learning rate
bias = np.random.rand(1)  # for now, set some random bias, can change later


for i in trainRoundList:
    print("myPerceptron training for {0} time(s)".format(i))

    # create the perceptron object
    myPerceptron = Perceptron(nFeatures, bias, lr)

    #print("Pre training weights: ")
    # print(myPerceptron.weights)

    # do the training
    currentError = myPerceptron.train(
        features, labels, i)  # train it by i times

    #print("Post training weights: ")
    # print(myPerceptron.weights)

    # test features
    testData = np.array([1, 0, 0])
    testOutput = myPerceptron.predict(testData)

    #print("Prediction for new data: ", testOutput)

    errorList.append(abs(currentError.mean()))  # save the mean of raw errors
    outputList.append(testOutput[0])

# plot to inspect the results
fig, axs = plt.subplots(2, figsize=(10, 8))
axs[0].plot(trainRoundList, outputList, 'b.-')
axs[0].set(ylabel="P(Class 1)")

axs[1].plot(trainRoundList, errorList, 'r*-')
plt.ylabel("Final prediction error")
plt.xlabel("Number of training rounds")
