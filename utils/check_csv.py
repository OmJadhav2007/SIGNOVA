
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

file_path = os.path.join(PROJECT_ROOT, "data", "Indian Sign Language Gesture Landmarks.csv")
if not os.path.exists(file_path):
    file_path = os.path.join(PROJECT_ROOT, "Indian Sign Language Gesture Landmarks.csv")
if not os.path.exists(file_path):
    file_path = "Indian Sign Language Gesture Landmarks.csv"

data = pd.read_csv(file_path)

print("Dataset shape:", data.shape)
print("\nColumn names:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head().to_string())

print("\nLast 5 columns:")
print(data.iloc[:, -5:].head().to_string())

print("\nData types:")
print(data.dtypes)

print("\nMissing values:", data.isnull().sum().sum())

print("\nPossible label columns:")
for col in data.columns:
    if any(word in col.lower() for word in ["label", "gesture", "sign", "class", "target", "gloss"]):
        print(col, "->", data[col].head().tolist())
