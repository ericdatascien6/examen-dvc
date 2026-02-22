import pandas as pd
from sklearn.model_selection import train_test_split
import os

# création du dossier processed_data si existe pas
os.makedirs("data/processed_data", exist_ok=True)

# chargement du dataset brut
df = pd.read_csv("data/raw/raw.csv")

# Separation target/ variables explicatives
X = df.drop(["silica_concentrate", "date"], axis=1)
y = df["silica_concentrate"]

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sauvegarde des datasets dans  data/processed_data
X_train.to_csv("data/processed_data/X_train.csv", index=False)
X_test.to_csv("data/processed_data/X_test.csv", index=False)
y_train.to_csv("data/processed_data/y_train.csv", index=False)
y_test.to_csv("data/processed_data/y_test.csv", index=False)
