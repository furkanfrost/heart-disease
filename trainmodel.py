# train_model.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Initialize and train the Random Forest model
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model

if __name__ == "__main__":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # Load the dataset directly from the correct path
    df = pd.read_csv("data/heart.csv")

    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train and evaluate
    model = train_and_evaluate(X_scaled, y)
