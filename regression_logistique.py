import numpy as np

# ===================== 1. Chargement des données =====================
def load_data():
    data_train = np.load('./data_train.npy', allow_pickle=True)
    data_test = np.load('./data_test.npy', allow_pickle=True)
    vocab_map = np.load('./vocab_map.npy', allow_pickle=True)
    
    label_train = np.genfromtxt('./label_train.csv', delimiter=',', dtype=int, skip_header=1, usecols=1) 

    return data_train, data_test, vocab_map, label_train

# ===================== 2. Normalisation et séparation des données =====================
def normalize_data(matrix):
    """Normalise les données en appliquant un z-score"""
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)

    # Éviter la division par 0
    return (matrix - mean) / (std + 1e-8)  

def split_data(X, y, test_size=0.2):
    """Sépare les données en deux ensembles manuellement"""
    split_idx = int(X.shape[0] * (1 - test_size))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    return X_train, X_val, y_train, y_val

# ===================== 3. Régression logistique =====================
def sigmoid(z):
    """Calcul de la fonction sigmoïde"""
    return 1 / (1 + np.exp(-z))

def compute_loss(X, y, w, b):
    """Calcul de la perte logistique"""
    m = X.shape[0]
    z = np.dot(X, w) + b
    predictions = sigmoid(z)
    loss = - (1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return loss

def gradient_descent(X, y, w, b, learning_rate, iterations):
    """Exécute la descente de gradient pour ajuster les poids"""
    m = X.shape[0]
    for i in range(iterations):
        z = np.dot(X, w) + b
        predictions = sigmoid(z)
        
        # Gradient des poids et du biais
        dw = (1 / m) * np.dot(X.T, (predictions - y))
        db = (1 / m) * np.sum(predictions - y)
        
        # Mise à jour des poids
        w -= learning_rate * dw
        b -= learning_rate * db
        
        if i % 100 == 0:
            loss = compute_loss(X, y, w, b)
            print(f"Iteration {i}, Loss: {loss}")
            
    return w, b

# ===================== 4. Entraînement et évaluation =====================
def train_model(X_train, y_train, X_val, y_val, learning_rate=0.01, iterations=1000):
    """Entraîne le modèle de régression logistique"""
    w = np.zeros(X_train.shape[1])
    b = 0
    
    # Entraînement
    w, b = gradient_descent(X_train, y_train, w, b, learning_rate, iterations)
    
    # Évaluation sur ensemble de validation
    val_loss = compute_loss(X_val, y_val, w, b)
    print(f"Validation Loss: {val_loss}")
    
    return w, b

# ===================== 5. Prédiction finale =====================
def predict(X, w, b):
    """Prédit les étiquettes des données d'entrée"""
    z = np.dot(X, w) + b
    return sigmoid(z) >= 0.5  # Retourne 1 si la probabilité est >= 0.5, sinon 0

# ===================== Fonction principale =====================
def main():
    # Chargement des données
    data_train, data_test, vocab_map, label_train = load_data()

    # Normalisation des données
    data_train_normalized = normalize_data(data_train)
    data_test_normalized = normalize_data(data_test)

    # Séparation des données en entraînement et validation
    X_train, X_val, y_train, y_val = split_data(data_train_normalized, label_train)

    # Entraînement du modèle
    w, b = train_model(X_train, y_train, X_val, y_val)

    # Prédiction finale sur l'ensemble de test
    test_predictions = predict(data_test_normalized, w, b)
    print("Test predictions:", test_predictions)

if __name__ == "__main__":
    main()
