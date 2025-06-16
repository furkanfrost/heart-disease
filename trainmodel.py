# train_model.py
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import numpy as np

def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Initialize and train the Decision Tree model
    model = DecisionTreeClassifier(random_state=random_state, max_depth=5)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model

def save_model_and_scaler(model, scaler, model_path="heart_model.pkl", scaler_path="heart_scaler.pkl"):
    """Save the trained model and scaler to disk."""
    try:
        # Ensure the model and scaler are using numpy arrays
        if hasattr(model, 'predict_proba'):
            test_input = np.zeros((1, model.n_features_in_))
            model.predict_proba(test_input)
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    except Exception as e:
        print(f"Error saving model files: {e}")
        raise

if __name__ == "__main__":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    try:
        # Load the dataset
        print("Loading dataset...")
        df = pd.read_csv("data/heart.csv")
        print("Dataset loaded successfully.")

        # Prepare features and target
        X = df.drop('target', axis=1)
        y = df['target']
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")

        # Scale features
        print("Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("Features scaled successfully.")

        # Train and evaluate
        print("Training model...")
        model = train_and_evaluate(X_scaled, y)
        print("Model trained successfully.")

        # Save model and scaler
        print("Saving model and scaler...")
        save_model_and_scaler(model, scaler)
        print("Model and scaler saved successfully.")

    except FileNotFoundError:
        print("Error: Could not find the heart.csv file in the data directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
