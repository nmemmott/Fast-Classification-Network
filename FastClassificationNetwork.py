import numpy as np
import sys

class FCNetwork:
    
    def _euclidDistance(self, point1, point2):
        if(len(point1) != len(point2)):
            raise ValueError("point1 and point2 must have the same dimensions (point1={}, point2={})".format(point1, point2))
        accum = 0
        for i in range(0, len(point1)):
            accum += (point1[i]-point2[i])**2
        return accum**0.5
    
    def _triangularMembership(self, distances, k):
        #Find the closest k distances
        kNearest = np.argsort(distances)[:k]
        #Find the denominator of the triangular membership function 
        denom = np.sum(1. / np.array([distances[i] for i in kNearest]))
        #Compute the membership grades
        memberGrades = np.empty(len(distances))
        testSum = 0
        for i in range(0,len(distances)):
            memberGrades[i] = (1./distances[i])/denom if i in kNearest else 0
        gradeTotal = np.sum(memberGrades)
        assert (gradeTotal>0.999 and gradeTotal<1.001), "The sum of the memberGrades ({}) is not equal to 1.0.".format(gradeTotal)
        return memberGrades
    
    def __init__(self, trainIn, trainOut, k):
        #TODO: Filter out duplicate samples 
        self.hiddenLength = len(trainIn)
        if self.hiddenLength == 0:
            raise valueError("There must be atleast one sample")
        if self.hiddenLength != len(trainOut):
            raise ValueError("Input length must equal output length")
        self.outLength = len(trainOut[0])
        self.inWeights = trainIn
        self.outWeights = np.array(trainOut)
        self.k = k
        self.radGen = np.full(self.hiddenLength, sys.maxsize, dtype=np.float32)
        #Find the radius of generalization for each hidden node
        for i in range(0, self.hiddenLength-1):
            for j in range(i+1, self.hiddenLength):
                #calculate distance between vector i and j
                dis = self._euclidDistance(trainIn[i], trainIn[j])/2
                #check if either is lower than their current value, replace it
                if dis < self.radGen[i]:
                    self.radGen[i] = dis
                if dis < self.radGen[j]:
                    self.radGen[j] = dis
        #Vector functions for the network
        # print("radGen = {}".format(self.radGen))
        self._actFunc = np.vectorize(lambda d, r: 0 if d <= r else d)
    
    def feedForward(self, input):
        #Calculate distance between each inWeight vector and the input
        d = [self._euclidDistance(self.inWeights[i], input) for i in range(0,self.hiddenLength)]
        #Get activation values
        h = self._actFunc(d, self.radGen)
        #Find a zero activation if any
        zInd = np.where(h == 0)[0]
        m = len(zInd)
        # assert (m<=1), "More than one activation was zero\n\n\tinput = {}\n\tzeroVectors = {}\n\tdistances = {}\n\tzInd = {}\n\tm = {}".format(input, [self.inWeights[i] for i in zInd], [d[i] for i in zInd], zInd, m)
        #Calculate the rule base output
        mu = np.empty(self.hiddenLength)
        if m>=1:
            #Use 1NN
            mu.fill(0)
            mu[zInd[0]] = 1
        else:
            #Use KNN
            mu = self._triangularMembership(d, self.k)
        # print("mu = {}".format(mu))
        #Do dot product between rule base output and output weights
        return np.array([np.dot(mu, self.outWeights[:,i]) for i in range(0, self.outLength)])
