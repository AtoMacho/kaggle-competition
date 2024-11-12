import numpy as np
import pandas as pd

# Loading each classes file into our program
class Load:
    def __init__(self, train, test, vocab, label):
        self.train = train
        self.test = test
        self.vocab = vocab
        self.label = label
    
    def load_data(self):
        data_train = np.load(self.train, allow_pickle=True)
        data_test = np.load(self.test, allow_pickle=True)
        vocab_map = np.load(self.vocab, allow_pickle=True)
        
        label_train = pd.read_csv(self.label, delimiter=',').iloc[:, 1].values  
        
        return data_train, data_test, vocab_map, label_train

# Processing the data by normalising, splitting and oversampling
class Data:
    def normalize_data(self, train_data , test_data):
        avg  = np.mean(train_data, axis=0)
        standard_dev = np.std(train_data, axis=0)
        normalized_train  = (train_data - avg ) / (standard_dev + 1e-8)
        normalized_test = (test_data - avg ) / (standard_dev + 1e-8)
        return normalized_train, normalized_test
    
    def split_data(self, data_train, label_train, test_size=0.2):
        idx = int(data_train.shape[0] * (1 - test_size))
        return data_train[:idx], data_train[idx:], label_train[:idx], label_train[idx:]
    
    def oversample_minority_class(self, X_train, y_train):
        X_minority = X_train[y_train == 1]
        y_minority = y_train[y_train == 1]
        
        X_balanced = np.concatenate([X_train, X_minority], axis=0)
        y_balanced = np.concatenate([y_train, y_minority], axis=0)
        
        return X_balanced, y_balanced

# Logistic regression model with L2 regularization
class LogisticRegressionModel:
    def __init__(self, learn_rate, iter, l2, threshold):
        self.learn_rate = learn_rate
        self.iter = iter
        self.l2 = l2
        self.threshold = threshold
        self.weight = None
        self.bias = 0
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, X, y):
        n_samples = X.shape[0]
        z = np.dot(X, self.weight) + self.bias
        predictions = self.sigmoid(z)
        loss = - (1 / n_samples) * np.sum(y * np.log(predictions + 1e-8) + (1 - y) * np.log(1 - predictions + 1e-8))
        l2_penalty = (self.l2 / (2 * n_samples)) * np.sum(self.weight**2)
        return loss + l2_penalty

    def compute_weights(self, y):
        class_counts = np.bincount(y.astype(int))
        total_samples = len(y)
        class_weights = total_samples / (len(class_counts) * class_counts)
        return class_weights

    def gradient_descent(self, X, y, class_weights):
        n_samples = X.shape[0]
        self.weight = np.zeros(X.shape[1])
        
        for i in range(self.iter):
            z = np.dot(X, self.weight) + self.bias
            predictions = self.sigmoid(z)
            weights = class_weights[y.astype(int)]
            
            weight_gradient = (1 / n_samples) * np.dot(X.T, weights * (predictions - y)) + (self.l2 / n_samples) * self.weight
            bias_gradient = (1 / n_samples) * np.sum(weights * (predictions - y))
            
            self.weight -= self.learn_rate * weight_gradient
            self.bias -= self.learn_rate * bias_gradient
            
            if i % 100 == 0:
                loss = self.compute_loss(X, y)
                print(f"Iteration {i}, Loss with L2: {loss}")

    def train(self, X, y):
        class_weights = self.compute_weights(y)
        self.gradient_descent(X, y, class_weights)
    
    def predict(self, X):
        z = np.dot(X, self.weight) + self.bias
        return self.sigmoid(z) >= self.threshold
    
    def compute_metrics(self, y_true, y_pred):
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        actual_positives = np.sum(y_true == 1)
        
        precision = true_positives / (predicted_positives + 1e-8)
        recall = true_positives / (actual_positives + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        
        return precision, recall, f1_score, accuracy

# main function
def main():
    # Parameters
    learn_rate = 0.001
    iterations = 1000
    lambda_l2 = 0.01
    threshold = 0.645
    
    # Load data
    load = Load('./data_train.npy', './data_test.npy', './vocab_map.npy', './label_train.csv')
    data_train, data_test, vocab_map, label_train = load.load_data()
    
    # Process data
    data = Data()
    data_train_normalized, data_test_normalized = data.normalize_data(data_train, data_test)
    X_train, X_val, y_train, y_val = data.split_data(data_train_normalized, label_train)
    X_train_balanced, y_train_balanced = data.oversample_minority_class(X_train, y_train)
    
    # Initialize model
    model = LogisticRegressionModel(learn_rate, iterations, lambda_l2, threshold)
    
    # Train model
    model.train(X_train_balanced, y_train_balanced)
    
    # Evaluate on validation set
    val_predictions = model.predict(X_val)
    precision, recall, f1_score, accuracy = model.compute_metrics(y_val, val_predictions)
    print(f"Validation Metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")
    
    # Predict on test set
    test_predictions = model.predict(data_test_normalized)
    output_df = pd.DataFrame({'ID': np.arange(len(test_predictions)), 'label': test_predictions.astype(int)})
    output_df.to_csv('submission.csv', index=False)
    print("Submission file 'submission.csv' created.")

if __name__ == "__main__":
    main()