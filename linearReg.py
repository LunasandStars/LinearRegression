import pandas
import matplotlib.pyplot as plt
import numpy

data = pandas.read_csv('MNIST_testHW2.csv')
dataTraining = pandas.read_csv('MNIST_trainingHW2.csv')

# Variables for testing and training data
trainingLabels = dataTraining.iloc[:, 0].as_matrix()   #class label
trainingData = dataTraining.drop('label', axis = 1).as_matrix()   #dropping the label in the data set and setting it to a matrix
testLabels = data.iloc[:, 0].as_matrix()
testData = data.drop('label', axis = 1).as_matrix()
# testData = numpy.array(data.drop('label', axis = 1))
# print(trainingLabels)
# print(trainingData)
# print(dataTraining)


#print(normalizeData(trainingData))

# Optimal Co-efficents using b_opt = (X'X)^(-1) * X'y (returns an array)
def b_opt(X, y):  #X is a matrix and y is a vector
    calculateb_opt = numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(X.transpose(), X)), X.transpose()), y)
    return calculateb_opt
# print("The Display of b_opt: ", b_opt(trainingData, trainingLabels))

# Classify the test data with threshold of 0.5 (Accuracy)
def accuracy(bestFitLine, testLabels):
    bestFitLine[bestFitLine > 0.5] = 1
    bestFitLine[bestFitLine <= 0.5] = 0
    val = sum(bestFitLine == testLabels) / float(len(testLabels)) * 100  # Number of times the bestFitLine is equal to testData divided by the number of testLabels
    return val

def prediction (X, b):  # This function will help with getting the accuracy after it does the dot product of testdata and testlabels
    # X = numpy.array(X)
    return numpy.array(numpy.dot(X, b))
# print(prediction(trainingData, b_opt(trainingData, trainingLabels)))

def linearRegression(X, y):
    return b_opt(X, y)

b_opt = linearRegression(trainingData,trainingLabels)
# print(b_opt)
pr = prediction(testData, b_opt)
acc = accuracy(pr, testLabels)
#Print b_opt and accurary
#>>>print "The b_opt is:", b_opt
#>>>print "The accuracy is:", acc, "%"


# Compute error from test data to line of regression
# Calculate the distance of the point and the predicted line
#Take the sum of the those distance and average it to get the error value


# Gradient Descent
# Normalize the data
def normalizeData(dataNorm):
    return dataNorm / 255

#This represents the learning rate, number of times you want it to iterate

def cost(X, y, b):
    return numpy.sum((numpy.dot(X, b) - numpy.array(y))**2)


def gradientDescent(X, y, b):
    # for index in range(len(X)):
        # X = normalizeData(X)
    return -numpy.dot(X.transpose(), y) + numpy.dot(numpy.dot(X.transpose(), X), b)
trainingData = normalizeData(trainingData)
testData = normalizeData(testData)
_, p = trainingData.shape
b_est = numpy.zeros(p)
learningRate = 1e-4
iterator = 100
bs = [b_est]
costs = [cost(trainingData, trainingLabels, b_est)]

for i in range(0, iterator):
    b_est = b_est - learningRate * gradientDescent(trainingData, trainingLabels, b_est)
    b_cost = cost(trainingData, trainingLabels, b_est)
    bs.append(b_est)
    costs.append(b_cost)

# check convergence
plt.plot(costs)
plt.show()

# plt.plot(bs, costs)
# plt.show()

# b_opt = b_est
# b_opt

# pr = gradientDescent(trainingData,trainingLabels, b_est)
# print(b_est)
pr = prediction(testData, b_est)
accuracyOfGD = accuracy(pr, testLabels)

print ("The accuracy of Gradient Descent is: ", accuracyOfGD, "%")