import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Veri setini okuma
data = pd.read_csv("Diagnosis of anemia with iron values\Datas\processed_data.csv")

# Hedef değişkeni ve bağımsız değişkenleri ayırma
X = data.drop("Result", axis=1)
y = data["Result"]

# Veri normalizasyonu (Önemli özellikle, özellikle farklı ölçeklerde özellikler varsa)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hiperparametreler için farklı değerler
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [2, 4, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search ile en iyi parametreleri bulma
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

# En iyi parametrelerle model oluşturma
best_rf = RandomForestClassifier(**best_params)
best_rf.fit(X_train, y_train)

# Modelin değerlendirilmesi
y_pred = best_rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))