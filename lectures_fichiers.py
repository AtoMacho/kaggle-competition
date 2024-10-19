import numpy as np

try:
    data_test = np.load('./data_test.npy', allow_pickle=True)
    data_train = np.load('./data_train.npy', allow_pickle=True)
    vocab_map = np.load('./vocab_map.npy', allow_pickle=True)
    
    print("Données chargées avec succès!")
    
except Exception as e:
    print(f"Erreur lors du chargement: {e}")