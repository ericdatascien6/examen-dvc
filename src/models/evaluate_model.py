import pandas as pd
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score

# chargement des données de test normalisées
X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv").values.ravel()

# Chargement du modèle entraînés
model = joblib.load("models/trained_model.pkl")

#  prédiction
y_pred = model.predict(X_test)

# métriques
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Sauvegarde des prédictions
predictions = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred
})
predictions.to_csv("data/predictions.csv", index=False)


#sauvegarde des scores
scores = {
    "mse": mse,
    "r2": r2
}

with open("metrics/scores.json", "w") as f:
    json.dump(scores, f)

