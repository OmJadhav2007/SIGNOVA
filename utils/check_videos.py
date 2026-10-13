
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

metadata_file = os.path.join(PROJECT_ROOT, "data", "metadata.csv")
if not os.path.exists(metadata_file):
    metadata_file = os.path.join(PROJECT_ROOT, "metadata.csv")
if not os.path.exists(metadata_file):
    metadata_file = "metadata.csv"

if not os.path.exists(metadata_file) or os.path.getsize(metadata_file) == 0:
    print(f"Metadata file '{metadata_file}' is empty or does not exist.")
    exit(0)

metadata = pd.read_csv(metadata_file)

print("Total records:", len(metadata))
print("\nAvailable columns:")
print(metadata.columns.tolist())

print("\nFirst 10 words:")
print(metadata["word"].head(10).to_string(index=False))

print("\nUnique word classes:", metadata["word"].nunique())
print("\nExample video paths:")
print(metadata["video_path"].head(5).to_string(index=False))
