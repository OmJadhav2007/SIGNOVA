# Signova: Real-Time Indian Sign Language (ISL) Interpreter

Signova is a computer vision and machine learning-powered system designed to recognize and interpret Indian Sign Language (ISL) hand gestures in real-time from a live webcam feed, converting them into English text.

---

## Project Structure

```text
signova/
├── data/
│   ├── isl_landmarks.csv         # 63-feature landmark dataset (Active training data)
│   ├── metadata.csv              # Placeholder/metadata for video datasets
│   └── (Indian Sign Language...csv) # External reference dataset (Ignored by Git >100MB)
├── models/
│   └── isl_model.pkl             # Trained Random Forest model & feature bundle
├── utils/
│   ├── check_csv.py              # Diagnostic script to inspect landmark datasets
│   └── check_videos.py           # Diagnostic script to inspect video metadata
├── collect_data.py               # Interactive webcam data collection tool
├── train_model.py                # Model training and evaluation script
├── isl_interpreter.py            # Live real-time webcam gesture interpreter
├── requirements.txt              # Project Python dependencies
├── .gitignore                    # Git ignore rules for environments & large files
└── README.md                     # Project documentation
```

---

## Supported Gestures

| Key (Collect) | Label | Output Word | Gesture Description |
| :---: | :---: | :---: | :--- |
| `1` | `SIGN_1` | **OKAY** | "OK" gesture (thumb and index finger forming a circle) |
| `2` | `SIGN_2` | **STOP** | Open flat palm facing the camera |
| `3` | `SIGN_3` | **THANKS** | Gratitude hand pose |
| `4` | `SIGN_4` | **PEACE** | Victory / Peace sign (V-sign) |
| `5` | `SIGN_5` | **GOOD** | Thumbs-up gesture |

---

## How It Works

1. **Hand Landmark Extraction**: Uses Google MediaPipe Hands to capture 21 3D hand landmarks ($21 \times 3 = 63$ features: $x, y, z$).
2. **Translation & Scale Normalization**:
   - **Wrist Centering**: Subtracts landmark 0 (wrist) coordinates from all landmarks (`coords - coords[0]`), ensuring the system is position-invariant across the webcam frame.
   - **Scale Invariance**: Normalizes coordinates by their maximum distance, ensuring accurate detection regardless of hand size or distance from the camera.
3. **Machine Learning Classification**: A `RandomForestClassifier` (300 estimators, balanced class weights) evaluates the 63 normalized coordinates and produces class probabilities.
4. **Confidence Gating & Temporal Smoothing**:
   - Predictions below **65% confidence** are filtered out.
   - A sliding 7-frame temporal window (`deque(maxlen=7)`) uses majority voting to prevent frame-to-frame flicker and ensure smooth text output.

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd signova
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Run Live Interpreter
To launch the real-time translation app:
```bash
python isl_interpreter.py
```
* **Controls**: Show gestures in front of the camera. Press `Q` in the webcam window to exit.

### 2. Collect Additional Data (Optional)
To record new gesture samples:
```bash
python collect_data.py
```
* **Controls**:
  * Press `1`-`5` to choose which gesture to record.
  * Hold the gesture in front of the camera.
  * Press `SPACE` to save the sample into `data/isl_landmarks.csv`.
  * Press `Q` to exit.

### 3. Train the Model
After collecting new samples, retrain the classifier:
```bash
python train_model.py
```
* Computes stratified validation accuracy and exports the updated bundle to `models/isl_model.pkl`.

---

## Note on Large Files
* The external `Indian Sign Language Gesture Landmarks.csv` dataset is ~113 MB and is excluded from Git via `.gitignore` because GitHub has a strict **100 MB** file limit.
* The core active dataset (`data/isl_landmarks.csv`) and lightweight pre-trained model (`models/isl_model.pkl`) are tracked, allowing the project to run out of the box after cloning.

