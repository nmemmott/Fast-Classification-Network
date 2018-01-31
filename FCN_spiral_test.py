import numpy as np
from FastClassificationNetwork import FCNetwork
import matplotlib.pyplot as plt


def printPattern(pattern):
    for y in range(0, len(pattern)):
        line = ''
        for x in range(0, len(pattern[y])):
            if round(pattern[y][x]) == 0:
                line = line + " "
            elif round(pattern[y][x]) == 1:
                line = line + "#"
            else:
                line = line + "!"
        print(line)

def createTrainingSamples(pattern, numberOfSamples):
    trainingSamples = np.empty([numberOfSamples,2])
    outputClass = np.empty([numberOfSamples,1])
    s = 0
    while s < numberOfSamples:
        #Randomly choose a coordinate
        x = np.random.randint(16)
        y = np.random.randint(16)
        #Prevent duplicate samples
        if [x,y] in trainingSamples.tolist():
            continue
        trainingSamples[s] = [x,y]
        outputClass[s] = pattern[y][x]
        s+=1
    return (trainingSamples, outputClass)

    

#Create Input and output data
spiral = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
          [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
          [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
          [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
          [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
printPattern(spiral)
(inputSamp, outputSamp) = createTrainingSamples(spiral, 32)

#Create network
spiralNetwork = FCNetwork(inputSamp, outputSamp, 3)

#Record predicted output and error
result = np.empty([16,16])
error = 0
for y in range(0,16):
    for x in range(0, 16):
        result[y,x] = spiralNetwork.feedForward([x,y])[0]
        error += (result[y,x] - spiral[y][x])**2
printPattern(result)
print("error = {}".format(error/(16**2)))

#Display results and matplotlib
oneSamps = np.array([inputSamp[i] for i in np.where(outputSamp==1)[0]])
zeroSamps = np.array([inputSamp[i] for i in np.where(outputSamp==0)[0]])

fig, (ax1, ax2) = plt.subplots(1, 2, sharex='all', sharey='all')
im1 = ax1.imshow(result, cmap=plt.get_cmap('hot'), interpolation='bilinear', vmin=np.amin(result), vmax=np.amax(result))
im2 = ax2.imshow(spiral, cmap=plt.get_cmap('hot'), interpolation='bilinear', vmin=np.amin(result), vmax=np.amax(result))
ax1.scatter(oneSamps[:,0],oneSamps[:,1])
ax1.scatter(zeroSamps[:,0],zeroSamps[:,1], c='r')
plt.show()
