import numpy as np
import pandas as pd

# Class to load and prepare data
class LoadData:
    def __init__(self, train_path, test_path, vocab_path, label_path):
        self.train_path = train_path
        self.test_path = test_path
        self.vocab_path = vocab_path
        self.label_path = label_path
    
    def load_data(self):
        data_train = np.load(self.train_path, allow_pickle=True)
        data_test = np.load(self.test_path, allow_pickle=True)
        vocab_map = np.load(self.vocab_path, allow_pickle=True)
        
        label_train = pd.read_csv(self.label_path, delimiter=',').iloc[:, 1].values  
        
        return data_train, data_test, vocab_map, label_train

# Class for preprocessing, including normalization and oversampling
class ProcessData:
    def normalize_data(self, train_matrix, test_matrix):
        mean = np.mean(train_matrix, axis=0)
        standard = np.std(train_matrix, axis=0)
        train_normalized = (train_matrix - mean) / (standard + 1e-8)
        test_normalized = (test_matrix - mean) / (standard + 1e-8)
        return train_normalized, test_normalized
    
    def split_data(self, X, y, test_size=0.2):
        split_idx = int(X.shape[0] * (1 - test_size))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
    
    def oversample_minority_class(self, X_train, y_train):
        X_minority = X_train[y_train == 1]
        y_minority = y_train[y_train == 1]
        
        X_train_balanced = np.concatenate([X_train, X_minority], axis=0)
        y_train_balanced = np.concatenate([y_train, y_minority], axis=0)
        
        return X_train_balanced, y_train_balanced

# Logistic regression model with L2 regularization
class LogisticRegressionModel:
    def __init__(self, learning_rate, iterations, lambda_l2, threshold):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_l2 = lambda_l2
        self.threshold = threshold
        self.weight = None
        self.bias = 0
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_loss_with_l2(self, X, y):
        n_samples = X.shape[0]
        z = np.dot(X, self.weight) + self.bias
        predictions = self.sigmoid(z)
        loss = - (1 / n_samples) * np.sum(y * np.log(predictions + 1e-8) + (1 - y) * np.log(1 - predictions + 1e-8))
        l2_penalty = (self.lambda_l2 / (2 * n_samples)) * np.sum(self.weight**2)
        return loss + l2_penalty

    def compute_class_weights(self, y):
        class_counts = np.bincount(y.astype(int))
        total_samples = len(y)
        class_weights = total_samples / (len(class_counts) * class_counts)
        return class_weights

    def gradient_descent(self, X, y, class_weights):
        n_samples = X.shape[0]
        self.weight = np.zeros(X.shape[1])
        
        for i in range(self.iterations):
            z = np.dot(X, self.weight) + self.bias
            predictions = self.sigmoid(z)
            weights = class_weights[y.astype(int)]
            
            weight_gradient = (1 / n_samples) * np.dot(X.T, weights * (predictions - y)) + (self.lambda_l2 / n_samples) * self.weight
            bias_gradient = (1 / n_samples) * np.sum(weights * (predictions - y))
            
            self.weight -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient
            
            if i % 100 == 0:
                loss = self.compute_loss_with_l2(X, y)
                print(f"Iteration {i}, Loss with L2: {loss}")

    def train(self, X, y):
        class_weights = self.compute_class_weights(y)
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

# Main function
def main():
    # Parameters
    learning_rate = 0.001
    iterations = 1000
    lambda_l2 = 0.01
    threshold = 0.645
    
    # Load data
    loader = LoadData('./data_train.npy', './data_test.npy', './vocab_map.npy', './label_train.csv')
    data_train, data_test, vocab_map, label_train = loader.load_data()
    
    # Process data
    preprocessor = ProcessData()
    data_train_normalized, data_test_normalized = preprocessor.normalize_data(data_train, data_test)
    X_train, X_val, y_train, y_val = preprocessor.split_data(data_train_normalized, label_train)
    X_train_balanced, y_train_balanced = preprocessor.oversample_minority_class(X_train, y_train)
    
    # Initialize model
    model = LogisticRegressionModel(learning_rate, iterations, lambda_l2, threshold)
    
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