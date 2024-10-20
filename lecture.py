import numpy as np
import sys

try:
    data_test = np.load('./data_test.npy', allow_pickle=True)
    data_train = np.load('./data_train.npy', allow_pickle=True)
    vocab_map = np.load('./vocab_map.npy', allow_pickle=True)

    #np.set_printoptions(threshold=sys.maxsize)
    
    print("Content of data_test.npy:")
    print(data_test)

    print("\nContent of data_train.npy:")
    print(data_train)

    print("\nContent of vocab_map.npy:")
    print(vocab_map)
    
except Exception as e:
    print(f"Erreur lors du chargement: {e}")