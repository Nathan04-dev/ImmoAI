import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib

# 1. Daten laden
df = pd.read_csv('train.csv')

# 2. Relevante Spalten auswählen & aufräumen (Genau in der Reihenfolge: Wohnfläche, Schlafzimmer, Badezimmer, Qualität)
spalten = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'OverallQual', 'SalePrice']
df = df[spalten].dropna()

X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'OverallQual']] # Features
y = df['SalePrice']                                             # Zielvariable

# 3. Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modell trainieren
model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# 5. Modell speichern
joblib.dump(model, 'immo_modell.pkl')
print("✅ Modell erfolgreich trainiert und als 'immo_modell.pkl' gespeichert!")