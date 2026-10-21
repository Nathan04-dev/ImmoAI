import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib


df = pd.read_csv('train.csv')


spalten = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'OverallQual', 'SalePrice']
df = df[spalten].dropna()

X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'OverallQual']] 
y = df['SalePrice']                                             


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)


joblib.dump(model, 'immo_modell.pkl')
print("✅ Modell erfolgreich trainiert und als 'immo_modell.pkl' gespeichert!")
