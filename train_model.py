
import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = (
    os.path.join(BASE_DIR, "data", "isl_landmarks.csv")
    if os.path.exists(os.path.join(BASE_DIR, "data", "isl_landmarks.csv"))
    else os.path.join(BASE_DIR, "isl_landmarks.csv")
)
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILE = os.path.join(MODEL_DIR, "isl_model.pkl")

# Load the collected landmark data
data = pd.read_csv(DATA_FILE)

if "label" not in data.columns:
    raise ValueError(
        "The CSV must contain a column named 'label'. "
        "Check isl_landmarks.csv."
    )

# The 63 MediaPipe coordinate features
features = [f"f{i}" for i in range(63)]

missing = [col for col in features if col not in data.columns]
if missing:
    raise ValueError(
        f"Missing landmark columns: {missing}. "
        "Your CSV format does not match this training code."
    )

data = data.dropna(subset=["label"] + features)

X = data[features].astype(float)
y = data["label"].astype(str)

print("Total samples:", len(data))
print("Gesture labels:")
print(y.value_counts().to_string())

if y.nunique() < 2:
    raise ValueError(
        "Collect samples for at least two different gestures "
        "before training."
    )

# Train a small test split only when there are enough samples
if y.value_counts().min() >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
else:
    print(
        "\nWarning: Some labels have fewer than 2 samples. "
        "Training on all available data without a test split."
    )
    X_train, X_test = X, None
    y_train, y_test = y, None

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

if X_test is not None:
    predictions = model.predict(X_test)
    print("\nTest accuracy:", round(
        accuracy_score(y_test, predictions), 3
    ))
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

# Train the final model on ALL saved samples
model.fit(X, y)

# Save model and expected feature order
joblib.dump(
    {
        "model": model,
        "features": features
    },
    MODEL_FILE
)

print("\nModel saved successfully to", MODEL_FILE)
print("Recognized labels:", list(model.classes_))
