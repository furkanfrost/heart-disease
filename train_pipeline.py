import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 'heart.csv')

if not os.path.exists(data_path):
    print(f"HATA: '{data_path}' bulunamadı. Lütfen data klasörüne heart.csv dosyasını ekleyin.")
    exit()

df = pd.read_csv(data_path)

X = df.drop('HeartDisease', axis=1) if 'HeartDisease' in df.columns else df.drop('target', axis=1)
y = df['HeartDisease'] if 'HeartDisease' in df.columns else df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [col for col in numeric_features if col in X.columns]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), [col for col in categorical_features if col in X.columns])
    ],
    remainder='passthrough' # Diğer sütunları (örn. FastingBS) olduğu gibi bırak
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("Model training")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nReport:")
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, 'heart_pipeline.pkl')
print("\nBaşarılı! 'heart_pipeline.pkl' dosyası oluşturuldu.")