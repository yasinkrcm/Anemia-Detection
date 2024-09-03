import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the dataset
data = pd.read_csv('Datas/anemia.csv')

# Separate the features (X) and the target variable (y)
X = data.drop('Result', axis=1)
y = data['Result']

# Apply Label Encoding to the 'Result' column
# This converts the categorical target variable into a numerical format (0 or 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Apply Min-Max Scaling
# Exclude 'Gender' from scaling
min_max_scaler = MinMaxScaler()
X_scaled = X.copy()  # Create a copy of X to apply scaling

# Loop through columns and apply Min-Max Scaling, except for 'Gender'
for col in X.columns:
    if col != "Gender":
        X_scaled[col] = min_max_scaler.fit_transform(X[[col]])

# Combine the results into a single DataFrame
# Add 'Gender' column back to the scaled data
processed_data = X_scaled
processed_data['Gender'] = X['Gender'].values

# Add the target variable back to the processed dataset
processed_data['Result'] = y_encoded

# Save the processed data to a new CSV file
processed_data.to_csv('Datas/processed_data.csv', index=False)

# The script ends here. The resulting dataset is saved as 'Datas/processed_data.csv'.
