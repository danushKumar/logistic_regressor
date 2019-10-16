import numpy as np
import pandas as pd 

from metrics import f1_score, accuracy, probability_to_preds
from algorithms import sigmoid

class model(object):
    
    def __init__(self, 
                train_x,
                train_y,
                valid_x, 
                valid_y,  
                bias = 0,
                weights = None,
                learning_rate = 0.007, 
                threshold = 0.5, 
                train = True):

        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.bias = bias
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.no_of_samples = train_x.shape[1]
        self.train = train

        if not weights:
            self.weights = np.zeros((train_x.shape[0], 1))
        else:
             self.weights = weights            


    def __call__(self, epoch):
        self.forward(epoch)

    def forward(self, epoch=10000):
        for i in range(epoch):
            print(f'epoch {i}')

            a = self.activation()
            preds = probability_to_preds(a, self.threshold)

            dz = a - self.train_y
            dw = np.dot(self.train_x, dz.T) / self.no_of_samples
            db = np.sum(dz) / self.no_of_samples

            self.optimizer(dw, db, self.learning_rate)
            
            acc = accuracy(a, self.train_y)
            f1 = f1_score(preds, self.train_y)
            
            print(f'f1_score {f1}')
            print(f'train accuracy {acc}')
            print(f'train loss {self.cost(self.train_y, a)}')

        self.validate()
            
    
    def optimizer(self, dw, db,learning_rate=0.001):
        self.weights = self.weights - (learning_rate * dw)
        self.bias = self.bias - (learning_rate * db)
    
    def validate(self):
        self.train = False
        
        a = self.activation()
        preds = probability_to_preds(a, self.threshold)
        
        acc = accuracy(a, self.valid_y)
        f1 = f1_score(preds, self.valid_y)

        print(f'f1 score {f1}')
        print(f'test accuracy {acc}')
    
    def activation(self):
        z = 0
        if self.train:
            z = np.dot(self.weights.T, self.train_x) + self.bias
        else: 
            z = np.dot(self.weights.T, self.valid_x) + self.bias
     
        a = sigmoid(z)
     
        return a
    
    def cost(self, actuals, prediction):
        cost = -1 * (np.dot(actuals, np.log(prediction).T) + np.dot((1 - actuals), np.log(1 - prediction).T)) / self.no_of_samples
        
        return cost
