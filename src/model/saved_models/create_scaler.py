import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Charger tes données d'entraînement (ou juste un exemple)
# Ici on suppose que X_train est un np.ndarray
X_train = np.load('C:\\Users\\yesmine\\Desktop\\Tanit\\src\\model\\saved_models\\X_train.npy')  # ou un autre chemin selon ton projet

# Créer le scaler et l'entraîner
scaler = StandardScaler()
scaler.fit(X_train)

# Sauvegarder le scaler
joblib.dump(scaler, 'C:\\Users\\yesmine\\Desktop\\Tanit\\src\\model\\saved_models\\scaler.pkl')

print("scaler.pkl créé avec succès !")
