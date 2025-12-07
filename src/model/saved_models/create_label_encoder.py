import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

classes = ['low', 'optimal', 'high']

# Créer le LabelEncoder et l’entraîner sur les classes
le = LabelEncoder()
le.fit(classes)

# Sauvegarder le fichier
joblib.dump(le, 'C:\\Users\\yesmine\\Desktop\\Tanit\\src\\model\\saved_models\\label_encoder.pkl')

print("label_encoder.pkl créé avec succès !")
