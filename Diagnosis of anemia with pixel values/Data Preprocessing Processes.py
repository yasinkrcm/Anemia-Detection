import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE

# Veriyi yükleyin
data = pd.read_csv('Datas/output.csv')

# Gender sütununu dönüştür
data['Gender'] = data['Gender'].map({'F': 0, 'M': 1, 'F ': 0, 'M ': 1})

# Anaemic sütununu encode et
label_encoder = LabelEncoder()
data['Anaemic'] = label_encoder.fit_transform(data['Anaemic'])

# One-hot encoding uygulama
one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
anaemic_encoded = one_hot_encoder.fit_transform(data[['Anaemic']])
anaemic_encoded_df = pd.DataFrame(anaemic_encoded, columns=one_hot_encoder.get_feature_names_out(['Anaemic']))

# Eski 'Anaemic' sütununu kaldır ve yeni sütunları ekle
data = pd.concat([data.drop(columns=['Anaemic']), anaemic_encoded_df], axis=1)

# MinMaxScaler uygulama
min_max_scaler = MinMaxScaler()
columns_to_scale = ['%Red Pixel', '%Green pixel', '%Blue pixel', 'Hb']
data[columns_to_scale] = min_max_scaler.fit_transform(data[columns_to_scale])

# Ortalama piksel, farklar ve logaritma hesaplama
data['Average Pixel'] = data[['%Red Pixel', '%Green pixel', '%Blue pixel']].mean(axis=1)
data['Red-Blue Difference'] = data['%Red Pixel'] - data['%Blue pixel']
data['Green-Red Difference'] = data['%Green pixel'] - data['%Red Pixel']
data['Red-Blue Difference'] = data['Red-Blue Difference'].abs()
data['Green-Red Difference'] = data['Green-Red Difference'].abs()

# MinMaxScaler'ı fark sütunlarına da uygulama
data[['Red-Blue Difference', 'Green-Red Difference']] = min_max_scaler.fit_transform(data[['Red-Blue Difference', 'Green-Red Difference']])

# Log Hb hesaplama
data['Log Hb'] = np.log(data['Hb'] + 1)

# "Log Hb" sütunundaki 0 değerleri düzeltme
min_positive_value = data[data['Log Hb'] > 0]['Log Hb'].min()
data['Log Hb'] = data['Log Hb'].replace(0, min_positive_value)

# Ek sütunlar ve risk hesaplamaları
data['Red-Blue Product'] = data['%Red Pixel'] * data['%Blue pixel']
data['Hb to Red Ratio'] = data['Hb'] / (data['%Red Pixel'] + 1)

# Risk hesaplaması için orijinal veriyi kullanma
original_data = pd.read_csv('Datas/output.csv')
original_data['Risk'] = np.where((original_data['Hb'] < 12) & (original_data['%Red Pixel'] > 0.7), 1, 0)

# Risk sütununu işlenmiş veriye ekleme
original_data.set_index('Number', inplace=True)  # Orijinal verideki 'Number' sütununu indeks olarak ayarlama
risk_mapping = original_data['Risk'].to_dict()
data['Risk'] = data['Number'].map(risk_mapping)

# Risk değerlerine normal değer olan 0.5'i ekleme
data['Risk'] = np.where(data['Risk'] == 0, 0.5, data['Risk'])

# Özellik ve hedef değişkeni ayır
X = data.drop(columns=['Gender'])
y = data['Gender']

# SMOTE ile veri dengeleme
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# X_smote ve y_smote'yi bir DataFrame'e dönüştür
X_smote_df = pd.DataFrame(X_smote, columns=X.columns)
y_smote_df = pd.DataFrame(y_smote, columns=['Gender'])

# X_smote ve y_smote'yi birleştir
balanced_data = pd.concat([X_smote_df, y_smote_df], axis=1)

# 'Number' sütununu ekleyin ve sıralı numaralandırma yapın
balanced_data['Number'] = range(1, len(balanced_data) + 1)

# 'Gender' sütununu 'Number' sütunundan sonra düzenleme
column_order = [
    'Number', 'Gender',
    '%Red Pixel', '%Green pixel', '%Blue pixel', 'Hb',
    'Average Pixel', 'Red-Blue Difference', 'Green-Red Difference',
    'Log Hb', 'Red-Blue Product', 'Hb to Red Ratio', 'Risk'
] + [col for col in balanced_data.columns if col.startswith('Anaemic')]

balanced_data = balanced_data[column_order]

# Veriyi kontrol etme
print(balanced_data.head())

# Sonuçları CSV olarak kaydet
balanced_data.to_csv('Datas/Balanced_Processed_Data.csv', index=False)

print("Güncellenmiş veri kaydedildi.")
 

# SMOTE İŞLEMİ SONRASI YENİ EKLENEN SATIRLARDA RISK VE ANAEMİC_1 SÜTUNLARINDA HATALAR TESPİT EDİLMİŞ OLUP EN YAKIN KOMŞUYA YUVARLANMIŞTIR!!