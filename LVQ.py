'''
William Johnson
'''

import LVQNet

def main():
    pValues = [[-3,2], [0,3], [3,2], [-3,-2], [0,-3], [3,-2]]
    categories = ['a', 'b', 'c', 'c', 'b', 'a']
    learningRate = 0.1
    trainingThreshold = 0.0000000001
    
    print("Training....")
    net = LVQNet.LVQNet(pValues, categories, learningRate)
    net.train(trainingThreshold)
    
    print("\nResults:")
    for weight in range(len(net.layer1)):
        print("Weight for " + str(pValues[weight]) + ": " + str(net.layer1[weight]))
    
if __name__ == '__main__':
    main()