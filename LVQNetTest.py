'''
William Ray Johnson
'''

import unittest
import LVQNet

class LVQNetTest(unittest.TestCase):
    def setUp(self):
        self.net = LVQNet.LVQNet([[-1,-1],[1,1],[1,-1],[-1,1]], [1,1,2,2], 0.1)
        
    def testCompete(self):
        self.net.layer1 = [[-1,-1],[1,1],[1,-1],[-1,1]]
        expectedWinner = 0
        actualWinner = self.net.compete([-1,-1])
        
        self.assertEqual(expectedWinner, actualWinner)

    def testIsCorrectlyCategorized(self):
        self.assertTrue(self.net.isCorrectlyCategorized(1, 0))
        
    def testIsCorrectlyCategorizedWithLetters(self):
        self.net.outputLayer = ['a', 'a', 'b', 'b']
        self.assertTrue(self.net.isCorrectlyCategorized(1, 0))
        
    def testRecalcuateLayer1Weight(self):
        newWeight = self.net.recalculateLayer1Weight(1, [-1,-1], 1)
        self.assertEqual(2, len(newWeight))
    
        
if __name__ == '__main__':
    unittest.main()