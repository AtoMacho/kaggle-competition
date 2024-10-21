import numpy as np
import pandas as pd
import sys

# ===================== Chargement des données =====================
def load_data():
    data_train = np.load('./data_train.npy', allow_pickle=True)
    data_test = np.load('./data_test.npy', allow_pickle=True)
    vocab_map = np.load('./vocab_map.npy', allow_pickle=True)
    
    label_train = pd.read_csv('./label_train.csv', delimiter=',')
    label_train = label_train.iloc[:, 1].values  

    return data_train, data_test, vocab_map, label_train



# ===================== Normalisation et séparation des données =====================
def normalize_data(train_matrix, test_matrix):
    """Normalise les données en appliquant un z-score"""
    mean = np.mean(train_matrix, axis=0)
    std = np.std(train_matrix, axis=0)
    train_normalized = (train_matrix - mean) / (std + 1e-8)
    test_normalized = (test_matrix - mean) / (std + 1e-8)
    return train_normalized, test_normalized

def split_data(X, y, test_size=0.2):
    """Sépare les données en deux ensembles manuellement"""
    split_idx = int(X.shape[0] * (1 - test_size))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

# ===================== Organisation des données déséquilibrées =====================
def oversample_minority_class(X_train, y_train):
    """Suréchantillonne la classe minoritaire pour équilibrer les classes"""
    # Sélection des exemples de la classe minoritaire
    X_minority = X_train[y_train == 1]
    y_minority = y_train[y_train == 1]
    
    # Réplication des exemples de la classe minoritaire pour équilibrer les classes
    X_train_balanced = np.concatenate([X_train, X_minority], axis=0)
    y_train_balanced = np.concatenate([y_train, y_minority], axis=0)
    
    return X_train_balanced, y_train_balanced



# ===================== Régression logistique avec perte L2 =====================
def sigmoid(z):
    """Calcul de la fonction sigmoïde"""
    return 1 / (1 + np.exp(-z))

def compute_loss_with_l2(X, y, w, b, lambda_l2):
    """Calcul de la perte logistique avec régularisation L2"""
    m = X.shape[0]
    z = np.dot(X, w) + b
    predictions = sigmoid(z)
    loss = - (1 / m) * np.sum(y * np.log(predictions + 1e-8) + (1 - y) * np.log(1 - predictions + 1e-8))
    l2_penalty = (lambda_l2 / (2 * m)) * np.sum(w**2)
    return loss + l2_penalty

def compute_class_weights(y):
    """Calcule les poids des classes en fonction de la fréquence des classes."""
    class_counts = np.bincount(y.astype(int))
    total_samples = len(y)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return class_weights

def gradient_descent(X, y, w, b, learning_rate, iterations, lambda_l2, class_weights, decay_rate = 0.1, epochs_drop=10):
    """Exécute la descente de gradient pour ajuster les poids avec régularisation L2"""
    m = X.shape[0]

    learning_rate_o = learning_rate

    for i in range(iterations):
        z = np.dot(X, w) + b
        predictions = sigmoid(z)
        
        # Gradient des poids et du biais avec régularisation L2
        # dw = (1 / m) * np.dot(X.T, (predictions - y)) + (lambda_l2 / m) * w
        # db = (1 / m) * np.sum(predictions - y)

        weights = class_weights[y.astype(int)]
        dw = (1 / m) * np.dot(X.T, weights * (predictions - y)) + (lambda_l2 / m) * w
        db = (1 / m) * np.sum(weights * (predictions - y))
        
        # Mise à jour des poids et du biais
        w -= learning_rate_o * dw
        b -= learning_rate_o * db
        
        if i % 100 == 0:
            loss = compute_loss_with_l2(X, y, w, b, lambda_l2)
            print(f"Iteration {i}, Loss with L2: {loss}")
        
        if i % epochs_drop == 0 and i > 0:
            learning_rate = step_decay(learning_rate, decay_rate, epochs_drop, i)

    return w, b

def step_decay(initial_lr, drop, epochs_drop, epoch):
    """Step decay function."""
    return initial_lr * (drop ** (epoch // epochs_drop))

# ===================== Calcul des métriques =====================
def compute_metrics(y_true, y_pred):
    """Calcule la précision, le rappel, le F1 score et l'accuracy"""
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    actual_positives = np.sum(y_true == 1)
    
    precision = true_positives / (predicted_positives + 1e-8)  # Éviter division par 0
    recall = true_positives / (actual_positives + 1e-8)  # Éviter division par 0
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    return precision, recall, f1_score, accuracy

# ===================== Prédictions avec seuil ajustable =====================
def predict_with_threshold(X, w, b, threshold):
    """Prédit les étiquettes des données d'entrée avec un seuil ajustable"""
    z = np.dot(X, w) + b

    # np.set_printoptions(threshold=sys.maxsize)
    # print(sigmoid(z))

    return sigmoid(z) >= threshold

def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """Simple grid search for hyperparameter tuning."""
    best_f1 = 0
    best_params = {}

    # Define the grid of hyperparameters to test
    learning_rates = [0.001, 0.01, 0.1]
    lambda_l2_values = [0.01, 0.1, 1.0]
    
    for learning_rate in learning_rates:
        for lambda_l2 in lambda_l2_values:
            print(f"Testing Learning Rate: {learning_rate}, L2 Regularization: {lambda_l2}")
            w, b = train_and_evaluate(X_train, y_train, X_val, y_val, learning_rate, iterations, lambda_l2, threshold)
            val_predictions = predict_with_threshold(X_val, w, b, threshold)
            _, _, f1_score, _ = compute_metrics(y_val, val_predictions)

            print(f"F1 Score: {f1_score}")

            if f1_score > best_f1:
                best_f1 = f1_score
                best_params = {'learning_rate': learning_rate, 'lambda_l2': lambda_l2}

    print(f"Best parameters: {best_params}, Best F1 Score: {best_f1}")
    return best_params
# ===================== Entraînement et évaluation =====================
def train_and_evaluate(X_train, y_train, X_val, y_val, learning_rate, iterations, lambda_l2, threshold):
    """Entraîne le modèle de régression logistique avec régularisation L2 et évalue avec un seuil ajustable"""
    w = np.zeros(X_train.shape[1])
    b = 0
    
    class_weights = compute_class_weights(y_train)

    # Entraînement
    w, b = gradient_descent(X_train, y_train, w, b, learning_rate, iterations, lambda_l2, class_weights)
    
    # Évaluation sur ensemble de validation
    val_loss = compute_loss_with_l2(X_val, y_val, w, b, lambda_l2)
    print(f"Validation Loss with L2: {val_loss}")
    
    # Prédictions sur l'ensemble de validation avec le seuil ajustable
    val_predictions = predict_with_threshold(X_val, w, b, threshold)
    
    # Calcul des métriques
    precision, recall, f1_score, accuracy = compute_metrics(y_val, val_predictions)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    
    return w, b

# ===================== Fonction principale =====================
def main():
    learning_rate = 0.001
    iterations = 1000
    lambda_l2 = 0.01
    threshold = 0.645

    k = 4

    print(f"Model Parameters:\n"
      f"Learning Rate: {learning_rate}\n"
      f"Number of Iterations: {iterations}\n"
      f"L2 Regularization Strength: {lambda_l2}\n"
      f"Decision Threshold: {threshold}")

    # Chargement des données
    data_train, data_test, vocab_map, label_train = load_data()

    # Normalisation des données
    data_train_normalized, data_test_normalized = normalize_data(data_train, data_test)

    # Séparation des données en entraînement et validation
    X_train, X_val, y_train, y_val = split_data(data_train_normalized, label_train)

    # Oversampling de la classe minoritaire sur l'ensemble d'entraînement
    X_train_balanced, y_train_balanced = oversample_minority_class(X_train, y_train)

    # best_params = hyperparameter_tuning(X_train_balanced, y_train_balanced, X_val, y_val)

    # print(best_params)

    # Entraînement et évaluation du modèle avec les paramètres définis
    w, b = train_and_evaluate(X_train_balanced, y_train_balanced, X_val, y_val, learning_rate, iterations, lambda_l2, threshold)

    # Prédiction finale sur l'ensemble de test avec le même seuil
    test_predictions = predict_with_threshold(data_test_normalized, w, b, threshold)
    print("Test predictions:", test_predictions)

    output_df = pd.DataFrame({
        'ID': np.arange(len(test_predictions)),  # Creating IDs from 0 to n-1
        'label': test_predictions.astype(int)  # Convert boolean predictions to int
    })

    output_df.to_csv('submission.csv', index=False)
    print("Submission file 'submission.csv' created.")

if __name__ == "__main__":
    main()
