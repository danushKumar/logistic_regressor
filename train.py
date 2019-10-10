import pickle

from data_loader import DataLoader
from classifier import ShallowLogisticClassifier

def train(path):
    data_loader = DataLoader(path)
    data_loader()
    classifier = ShallowLogisticClassifier(data_loader.train_x,
                                           data_loader.train_y,
                                           data_loader.valid_x,
                                           data_loader.valid_y, 
                                           threshold = 0.4,
                                           learning_rate = 0.005)
    classifier(epoch = 10000)
    save_model(classifier)

def save_model(model):
    pkl_representation = pickle.dumps(model)
    
    with open('model', 'wb') as file:
         
        pickle.dump(pkl_representation, file) 

if __name__=='__main__':
    train('data.csv')