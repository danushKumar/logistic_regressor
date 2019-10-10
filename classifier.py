import numpy as np
import pandas as pd 

from metrics import f1_score, accuracy
from algorithms import sigmoid

class ShallowLogisticClassifier(object):

    def __init__(self, train_x, train_y, valid_x, valid_y,  bias = 0, learning_rate = 0.007, threshold = 0.5, train=True):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.w = np.zeros((train_x.shape[0], 1))
        self.bias = bias
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.no_of_samples = train_x.shape[1]
        self.train = train

    def __call__(self, epoch = 5000):
        
        self.optimize(epoch)

    def propogate(self):

        # Z = np.dot(self.w.T, self.train_x) 
        A = self.activation()
        preds = self.probability_to_preds(A)

        dz = A - self.train_y
        dw = np.dot(self.train_x, dz.T) / self.no_of_samples
        db = (1 / self.no_of_samples) * np.sum(dz)

        cost = self.cost(A, self.train_y)
        f1 = f1_score(preds, self.train_y)
        
        print(f'cost = {cost}')
        print(f'f1 score = {f1}')
        print(f'accuracy = {accuracy(A, self.train_y)}')

        return {'dw': dw, 'db': db}, cost
    
    def optimize(self, epoch):

        self.cost_list = []

        for i in range(epoch):
            print(f'epoch :{i}')
            
            grads, cost = self.propogate()

            self.w = self.w - (self.learning_rate * grads['dw'])
            self.bias = self.bias - (self.learning_rate * grads['db'])

            if i % 100 == 0:
                self.cost_list.append(cost)

        self.validate()

    def probability_to_preds(self, a):

        preds = a >= self.threshold

        return preds.astype(int)
    

    def validate(self):
        
        self.train = False

        A = self.activation()
        preds = self.probability_to_preds(A)
        f1 = f1_score(preds, self.valid_y)

        print(f'valid f1_score {f1}')
        print(f'valid acc {accuracy(A, preds)}')

    def activation(self):
        
        z = 0

        if self.train:
            z = np.dot(self.w.T, self.train_x)
        else: 
            z = np.dot(self.w.T, self.valid_x)
        
        a = sigmoid(z)
        
        return a

    def cost(self, preds, actuals):
        print(preds.shape, actuals.shape)
        print(self.no_of_samples)
        cost = -1 * (np.dot(actuals, np.log(preds).T) + np.dot((1 - actuals), np.log(1 - preds).T)) / self.no_of_samples
        
        return cost
