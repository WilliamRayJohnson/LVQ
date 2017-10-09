'''
William Ray Johnson
'''

import math
import random

class LVQNet():
    def __init__(self, pValues, inputs, inputCategories, outputLayer, learningRate):
        self.inputs = inputs
        self.inputCategories = inputCategories
        self.outputLayer = outputLayer
        self.learningRate = learningRate
        self.layer1 = []
        for pValue in pValues:
            node = []
            for value in pValue:
                node.append(random.random())
            self.layer1.append(node)
        
    def compete(self, input):
        smallestDistance = float("inf")
        winnerIndex = 0
        for node in self.layer1:
            distance = self.distanceFormula(input, node)
            if distance < smallestDistance:
                smallestDistance = distance
                winnerIndex = self.layer1.index(node)
                
        return winnerIndex
        
    def distanceFormula(self, pointOne, pointTwo):
        a = pointOne[0] - pointTwo[0]
        b = pointOne[1] - pointTwo[1]
        
        return math.sqrt(a**2 + b**2)
        
    def isCorrectlyCategorized(self, winnerIndex, pIndex):
        return self.outputLayer[winnerIndex] == self.inputCategories[pIndex]
        
    def recalculateLayer1Weight(self, movementDirection, input, winnerIndex):
        newWeight = []
        for valueIndex in range(len(input)):
            newValue = (self.layer1[winnerIndex][valueIndex] + movementDirection * self.learningRate * 
                            (input[valueIndex] - self.layer1[winnerIndex][valueIndex]))
            newWeight.append(newValue)
            
        return newWeight
        
    def train(self, movementThreshold):
        thresholdMet = False
        while not thresholdMet:
            for pIndex in range(len(self.inputs)):
                winnerIndex = self.compete(self.inputs[pIndex])
                correctlyCategorized = self.isCorrectlyCategorized(winnerIndex, pIndex)
                if correctlyCategorized:
                    newWeight = self.recalculateLayer1Weight(1, self.inputs[pIndex], winnerIndex)
                else:
                    newWeight = self.recalculateLayer1Weight(-1, self.inputs[pIndex], winnerIndex)
                weightDistance = self.distanceFormula(self.layer1[winnerIndex], newWeight)
                if weightDistance <= movementThreshold:
                    thresholdMet = True
                else:
                    thresholdMet = False
                    
                self.layer1[winnerIndex] = newWeight