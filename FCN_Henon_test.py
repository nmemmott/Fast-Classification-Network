import matplotlib.pyplot as plt
import numpy as np

from FastClassificationNetwork import FCNetwork


WINDOW_SIZE = 4
SAMPLE_SIZE = 450
TEST_SIZE = 50
A = 1.4
B = 0.3

def HenonMap(xm1, xm2):
    return 1. - A*(xm1**2.) + B*(xm2**2.)

    
#Get points of data
dataPoints = np.empty(WINDOW_SIZE + SAMPLE_SIZE + TEST_SIZE)
#Set initial conditions
dataPoints[0] = 0.3
dataPoints[1] = 0.1
for i in range(2, WINDOW_SIZE + SAMPLE_SIZE + TEST_SIZE):
    dataPoints[i] = HenonMap(dataPoints[i-1], dataPoints[i-2])

#Assign data points to our sample and test arrays
inputSamp = np.empty((SAMPLE_SIZE, WINDOW_SIZE))
outputSamp = np.empty((SAMPLE_SIZE,1))
testData = np.empty((TEST_SIZE, WINDOW_SIZE))
testOut = np.empty(TEST_SIZE)
for i in range(0, SAMPLE_SIZE + WINDOW_SIZE + TEST_SIZE):
    #Assign sample data
    if i < SAMPLE_SIZE:
        for j in range(0,WINDOW_SIZE):
            inputSamp[i][j] = dataPoints[i+j]
    if i >= WINDOW_SIZE and i < SAMPLE_SIZE+WINDOW_SIZE:
        outputSamp[i-WINDOW_SIZE][0] = dataPoints[i]
    #Assign test data
    if i >= SAMPLE_SIZE and i < SAMPLE_SIZE + TEST_SIZE:
        for j in range(0,WINDOW_SIZE):
            testData[i-SAMPLE_SIZE][j] = dataPoints[i+j]
    if i >= SAMPLE_SIZE + WINDOW_SIZE:
        testOut[i-SAMPLE_SIZE-WINDOW_SIZE] = dataPoints[i]

#Create network and get network test output
henonNetwork = FCNetwork(inputSamp, outputSamp, 5)
netOut = np.empty(TEST_SIZE)
error = 0
for i in range(0, TEST_SIZE):
    netOut[i] = henonNetwork.feedForward(testData[i])
    error = (netOut[i]-testOut[i])**2
print("error = {}".format(error))

#Plot predicted and actual output
# plt.scatter(dataPoints[1:WINDOW_SIZE + SAMPLE_SIZE + TEST_SIZE],dataPoints[0:WINDOW_SIZE + SAMPLE_SIZE + TEST_SIZE-1])
plt.plot(np.arange(0,TEST_SIZE), netOut, color='r', marker='o');
plt.plot(np.arange(0,TEST_SIZE), testOut, color='b', marker='o');
plt.show()
