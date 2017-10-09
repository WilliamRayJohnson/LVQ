'''
William Johnson
'''

import random
import LVQNet

def main():
    pValues = [[-3,2], [0,3], [3,2], [-3,-2], [0,-3], [3,-2]]
    outputLayer = ['a', 'b', 'c', 'c', 'b', 'a']
    learningRate = 0.1
    trainingThreshold = 0.15
    valuesPerNode = 50
    testValues = 5
    inputs = []
    inputCategories = []
    testInputs = []
    testInputCategories = []
    
    for pValue in pValues:
        for input in range(valuesPerNode):
            newInput = []
            for value in pValue:
                newInput.append(random.uniform(value - 2, value + 2))
            inputs.append(newInput)
            inputCategories.append(outputLayer[pValues.index(pValue)])
            
    print("Training....")
    net = LVQNet.LVQNet(pValues, inputs, inputCategories, outputLayer, learningRate)
    net.train(trainingThreshold)
    
    for pValue in pValues:
        for input in range(testValues):
            newInput = []
            for value in pValue:
                newInput.append(random.uniform(value - 2, value + 2))
            testInputs.append(newInput)
            testInputCategories.append(outputLayer[pValues.index(pValue)])
    
    print("\nResults:")
    for weight in range(len(net.layer1)):
        print("Weight for " + str(pValues[weight]) + ": " + str(net.layer1[weight]))
    
if __name__ == '__main__':
    main()