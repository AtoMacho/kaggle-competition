import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler

# Loading each classes file into our program
def load_data():
    data_train = np.load('data_train.npy', allow_pickle=True)
    data_test = np.load('data_test.npy', allow_pickle=True)
    label_train = pd.read_csv('label_train.csv', delimiter=',').iloc[:, 1].values  
    return csr_matrix(data_train), csr_matrix(data_test), label_train

# Processing the data by normalising and reduction of dimensions
def preprocess_data(data_train, data_test):
    # Reduce dimensions with TruncatedSVD
    svd = TruncatedSVD(n_components=550)
    train_reduced = svd.fit_transform(data_train)
    test_reduced = svd.transform(data_test)
    
    #normalise the data
    scaler = StandardScaler()
    normalized_train = scaler.fit_transform(train_reduced)
    normalized_test = scaler.transform(test_reduced)
    
    return normalized_train, normalized_test

#find the best threshold of the model
def find_best_threshold(model, X_val, y_val):
    val_probabilities = model.predict_proba(X_val)[:, 1]
    best_threshold = 0.5
    best_f1 = 0
    for threshold in np.arange(0.45, 0.8, 0.005):
        val_predictions = (val_probabilities >= threshold).astype(int)
        f1 = f1_score(y_val, val_predictions)
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    print(f"Best Threshold: {best_threshold} with F1 Score: {best_f1}")
    return best_threshold

# Train data with different parameters
def train_and_evaluate(data_train, label_train):
    X_train, X_val, y_train, y_val = train_test_split(data_train, label_train, test_size=0.2, random_state=42)
    best_f1 = 0
    best_params = {}

    C_values = [0.001, 0.01, 0.1, 1]
    class_weights = [None, 'balanced']

    for C in C_values:
        for class_weight in class_weights:
            model = LogisticRegression(C=C, class_weight=class_weight, max_iter=3000, solver='saga',penalty='l2')
            model.fit(X_train, y_train)
            val_predictions = model.predict(X_val)
            f1 = f1_score(y_val, val_predictions)

            print(f"C: {C}, class_weight: {class_weight}, F1 Score: {f1}")

            if f1 > best_f1:
                best_f1 = f1
                best_params = {'C': C, 'class_weight': class_weight}

    print("\nBest parameters :", best_params)
    print("Best F1 score with the best_params :", best_f1)

    # train the model
    best_model = LogisticRegression(**best_params, max_iter=3000, solver='saga',penalty='l2')
    best_model.fit(data_train, label_train)
    
    best_threshold = find_best_threshold(best_model, X_val, y_val)

    return best_model, best_threshold

# Prediction of the best model with the threshold found
def save_predictions(model, data_test, best_threshold):
    threshold = best_threshold

    test_probabilities = model.predict(data_test)
    test_predictions = (test_probabilities >= threshold).astype(int)
    output = pd.DataFrame({'Id': range(len(test_predictions)), 'Label': test_predictions})
    output.to_csv('submission_improved.csv', index=False)
    print("Predictions saved into submission_improved.csv")

#main function
if __name__ == "__main__":
    data_train, data_test, label_train = load_data()
    data_train, data_test = preprocess_data(data_train, data_test)
    best_model, best_threshold = train_and_evaluate(data_train, label_train)
    save_predictions(best_model, data_test, best_threshold)
