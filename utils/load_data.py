import pickle

def load_data(data_path):
    
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)
        
    return data