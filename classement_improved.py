import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler

# ===================== Chargement des Données =====================
def load_data():
    data_train = np.load('data_train.npy', allow_pickle=True)
    data_test = np.load('data_test.npy', allow_pickle=True)
    label_train = pd.read_csv('label_train.csv', delimiter=',').iloc[:, 1].values  
    return csr_matrix(data_train), csr_matrix(data_test), label_train

# ===================== Prétraitement =====================
def preprocess_data(data_train, data_test):
    # Réduction de dimension avec TruncatedSVD pour données creuses
    svd = TruncatedSVD(n_components=100)  # Ajustez n_components selon les performances
    data_train_reduced = svd.fit_transform(data_train)
    data_test_reduced = svd.transform(data_test)
    
    # Normalisation des données réduites
    scaler = StandardScaler()
    data_train_normalized = scaler.fit_transform(data_train_reduced)
    data_test_normalized = scaler.transform(data_test_reduced)
    
    return data_train_normalized, data_test_normalized

# ===================== Entraînement et Évaluation =====================
def train_and_evaluate(data_train, label_train):
    X_train, X_val, y_train, y_val = train_test_split(data_train, label_train, test_size=0.2, random_state=42)
    best_f1 = 0
    best_params = {}

    # Paramètres à tester
    C_values = [0.1, 1, 10, 100]
    class_weights = [None, 'balanced']

    for C in C_values:
        for class_weight in class_weights:
            model = LogisticRegression(C=C, class_weight=class_weight, max_iter=1000, solver='liblinear')
            model.fit(X_train, y_train)
            val_predictions = model.predict(X_val)
            f1 = f1_score(y_val, val_predictions)

            # Affichage des paramètres et du score F1 pour chaque itération
            print(f"Iteration - C: {C}, class_weight: {class_weight}, F1 Score: {f1}")

            # Garder en mémoire les meilleurs paramètres
            if f1 > best_f1:
                best_f1 = f1
                best_params = {'C': C, 'class_weight': class_weight}

    print("\nMeilleurs paramètres trouvés :", best_params)
    print("Meilleur score F1 sur validation :", best_f1)

    # Entraîner le modèle final avec les meilleurs paramètres
    best_model = LogisticRegression(**best_params, max_iter=1000, solver='liblinear')
    best_model.fit(data_train, label_train)
    
    return best_model

# ===================== Prédiction et Sauvegarde =====================
def save_predictions(model, data_test):
    test_predictions = model.predict(data_test)
    output = pd.DataFrame({'Id': range(len(test_predictions)), 'Label': test_predictions})
    output.to_csv('submission_improved.csv', index=False)
    print("Prédictions sauvegardées dans submission_improved.csv")

# ===================== Fonction Principale =====================
if __name__ == "__main__":
    data_train, data_test, label_train = load_data()
    data_train, data_test = preprocess_data(data_train, data_test)
    best_model = train_and_evaluate(data_train, label_train)
    save_predictions(best_model, data_test)
