import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load des  données normalisées
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

#chargement des meilleurs paramètres du  GridSearch
best_params = joblib.load("models/best_params.pkl")

# initialisation modèle
model = RandomForestRegressor(**best_params, random_state=42)

# entraînement
model.fit(X_train, y_train)

# sauvegarde du  modèle
joblib.dump(model, "models/trained_model.pkl")

