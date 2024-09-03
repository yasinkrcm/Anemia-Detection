import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from scipy import stats
from imblearn.over_sampling import SMOTE

data = pd.read_csv("Datas/output.csv")

def categorize_hb(hb_value):
    if hb_value < 12:
        return 0
    elif 12 <= hb_value <= 16:
        return 0.5
    else:
        return 1
    
data['Gender'] = data['Gender'].map({'F': 0, 'M': 1, 'F ':0 , 'M ':1})
data['Hb Category'] = data['Hb'].apply(lambda x: categorize_hb(x))
data['Red-Blue Product'] = data['%Red Pixel'] * data['%Blue pixel']
data['Hb to Red Ratio'] = data['Hb'] / (data['%Red Pixel'] + 1)
data['Risk'] = np.where((data['Hb'] < 12) & (data['%Red Pixel'] > 0.7), 1, 0)

label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Anaemic_1'] = label_encoder.fit_transform(data['Anaemic'])

one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
anaemic_encoded = one_hot_encoder.fit_transform(data[['Anaemic']])
anaemic_encoded_df = pd.DataFrame(anaemic_encoded, columns=one_hot_encoder.get_feature_names_out(['Anaemic']))
data = pd.concat([data, anaemic_encoded_df], axis=1)
data.drop(columns=['Anaemic'], inplace=True)

min_max_scaler = MinMaxScaler()
columns_to_scale = ['%Red Pixel', '%Green pixel', '%Blue pixel', 'Hb']
data[columns_to_scale] = min_max_scaler.fit_transform(data[columns_to_scale])

data['Average Pixel'] = data[['%Red Pixel', '%Green pixel', '%Blue pixel']].mean(axis=1)
data['Red-Blue Difference'] = data['%Red Pixel'] - data['%Blue pixel']
data['Green-Red Difference'] = data['%Green pixel'] - data['%Red Pixel']
data['Log Hb'] = np.log(data['Hb'] + 1)


print(data.isnull().sum())

if data.isnull().sum().sum() == 0:
    np.random.seed(42)
    for col in data.columns:
        data.loc[data.sample(frac=0.05).index, col] = np.nan

print(data.isnull().sum())

data.to_csv('Datas/data_with_missing.csv', index=False)

missing_columns = data.columns[data.isnull().any()]
for column in missing_columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        data[column].fillna(data[column].mean(), inplace=True)

print(data.isnull().sum())

data.to_csv('Datas/data_filled.csv', index=False)

z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
threshold = 3
outliers = (z_scores > threshold).sum(axis=1)

data_no_outliers = data[(z_scores < threshold).all(axis=1)]

data_capped = data.copy()
for column in data.select_dtypes(include=[np.number]).columns:
    upper_limit = data[column].mean() + 3 * data[column].std()
    lower_limit = data[column].mean() - 3 * data[column].std()
    data_capped[column] = np.clip(data[column], lower_limit, upper_limit)

data_capped.to_csv('Datas/OutputNotOutliers.csv', index=False)

X = data.drop(columns=['Gender'])
y = data['Gender']

print(np.unique(y))
