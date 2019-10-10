from data_loader import DataLoader
from classifier import ShallowLogisticClassifier
import pickle

def train(path):
    data_loader = DataLoader(path)
    data_loader()
    classifier = ShallowLogisticClassifier(data_loader.train_x,
                                           data_loader.train_y,
                                           data_loader.valid_x,
                                           data_loader.valid_y)
    classifier()
    save_model(classifier)

def save_model(model):
    with open('model', 'wb') as file:
        pickle.dump(model, file) 

if __name__=='__main__':
    train('data.csv')