import pickle

from data_loader import DataLoader
from classifier import model

def train(data_path, model_file):
    data_loader = DataLoader(data_path)
    data_loader()

    weights = load_weights(model_file) if model_file else None

    classifier = model(data_loader.train_x,
                                           data_loader.train_y,
                                           data_loader.valid_x,
                                           data_loader.valid_y, 
                                           threshold = 0.4,
                                           weights = weights,
                                           learning_rate = 0.003)
    classifier(epoch = 10000)
    save_model(classifier)

def save_model(model):
    pkl_representation = pickle.dumps(model)
    
    with open('model', 'wb') as file:
         
        pickle.dump(pkl_representation, file) 

def load_weights(path):
    pkl_obj = None
    model_obj = None

    with open(path, 'rb') as file:
        pkl_obj = pickle.load(file)

    if pkl_obj:
        model_obj = pickle.loads(pkl_obj)

    return model_obj.weights

if __name__=='__main__':
    train(data_path = 'data.csv', model_file = 'model')
