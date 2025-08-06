# save_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
df = pd.read_csv(os.path.join('data', 'heart.csv'))

# Split into features and target
X = df.drop('target', axis=1)
y = df['target']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Ensure models directory exists
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Save model and scaler
joblib.dump(model, os.path.join(MODEL_DIR, 'heart_model.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'heart_scaler.pkl'))

print("Model and scaler have been saved successfully.")
