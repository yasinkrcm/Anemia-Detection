import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Veriyi yükle
data = pd.read_csv('Datas/guncellenmis_data.csv')

# Özellikler ve hedef değişkeni ayır
X = data.drop(columns=['Anaemic_1'])  # 'Anaemic' sütunu hedef değişken olarak ayrılır
y = data['Anaemic_1']

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Özellikleri ölçeklendir
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelleri tanımla
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Modelleri eğit ve değerlendir
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Results for {model_name}:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("="*50)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Random Forest modelini oluştur
rf_model = RandomForestClassifier(random_state=42)

# K-katlı çapraz doğrulama ile modelin performansını değerlendirin
cv_scores = cross_val_score(rf_model, X, y, cv=5)

# Çapraz doğrulama sonuçlarını yazdır
print("Çapraz Doğrulama Doğruluk Skorları: ", cv_scores)
print("Ortalama Doğruluk: ", cv_scores.mean())
