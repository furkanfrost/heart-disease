# data_preprocessing.py
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_column='target'):
    """
    - df: pandas DataFrame
    - target_column: string, name of the target column
    Returns:
      X_scaled: np.ndarray, scaled feature matrix
      y: pandas Series, target vector
      scaler: StandardScaler, fitted scaler object
    """
    # Debug: show existing columns
    print("Columns in DataFrame:", df.columns.tolist())

    # Feature/target split
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame columns.")
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

if __name__ == "__main__":
    # Determine paths relative to this file’s location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'heart.csv')

    # Load data
    df = pd.read_csv(csv_path)

    # Preprocess
    X_scaled, y, scaler = preprocess_data(df, target_column='target')
    print("Preprocessing complete. X_scaled shape:", X_scaled.shape)
