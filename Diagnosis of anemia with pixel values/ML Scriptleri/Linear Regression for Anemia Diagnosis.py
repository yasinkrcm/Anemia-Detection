import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv("Datas\guncellenmis_data.csv")
scaler = MinMaxScaler()

data[['%Red Pixel', '%Green pixel', '%Blue pixel']] = scaler.fit_transform(data[['%Red Pixel', '%Green pixel', '%Blue pixel']])

X = pd.get_dummies(data[['%Red Pixel', '%Green pixel', '%Blue pixel', 'Gender']], drop_first=True)

data = data.dropna()

X = data[['%Red Pixel', '%Green pixel', '%Blue pixel', 'Gender']]
y = data['Hb']

X['Gender'] = X['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Ortalama Karesel Hata (MSE): {mse}")
print(f"R^2 Skoru: {r2}")

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue', label='Tahminler')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Doğru Çizgi')
plt.xlabel('Gerçek Hb Değerleri')
plt.ylabel('Tahmin Edilen Hb Değerleri')
plt.title('Gerçek vs Tahmin Edilen Hb Değerleri')
plt.legend()
plt.show()

print("Özellik Katsayıları:", lr_model.coef_)
print("Sabit Terim:", lr_model.intercept_)
