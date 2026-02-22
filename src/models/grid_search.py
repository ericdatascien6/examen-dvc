import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import os

# Création du dossier models si existe pas
os.makedirs("models", exist_ok=True)



# chargement des données  (normalisées)
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

# init du modèle
model = RandomForestRegressor(random_state=42)

#  dict d'hyperparamètres
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20]
}

#gridSearch validation croisée
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

#fit du GridSearch
grid_search.fit(X_train, y_train)

# sauvegarde des best paramètres
joblib.dump(grid_search.best_params_, "models/best_params.pkl")
