
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
from collections import deque, Counter

# -----------------------------
# SETTINGS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = (
    os.path.join(BASE_DIR, "models", "isl_model.pkl")
    if os.path.exists(os.path.join(BASE_DIR, "models", "isl_model.pkl"))
    else os.path.join(BASE_DIR, "isl_model.pkl")
)
CONFIDENCE_THRESHOLD = 0.65
SMOOTHING_FRAMES = 7

# Your gesture-to-text mapping
LABEL_TO_TEXT = {
    "SIGN_1": "OKAY",
    "SIGN_2": "STOP",
    "SIGN_3": "THANKS",
    "SIGN_4": "PEACE",
    "SIGN_5": "GOOD"
}

# -----------------------------
# LOAD MODEL
# -----------------------------
bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
FEATURES = bundle["features"]

if len(FEATURES) != 63:
    raise ValueError(
        f"Expected 63 features, but model uses {len(FEATURES)}."
    )

print("Signova model loaded!")
print("Recognized labels:", list(model.classes_))
print("Press Q in the webcam window to quit.")

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    hands.close()
    raise RuntimeError(
        "Cannot open webcam. Close other camera applications "
        "and check your camera connection."
    )

recent_predictions = deque(maxlen=SMOOTHING_FRAMES)
display_text = "Show a hand gesture"
confidence = 0.0

try:
    while True:
        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            # Draw detected hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract 21 landmarks x (x, y, z) = 63 features
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand.landmark],
                dtype=np.float32
            )

            # Normalize relative to wrist
            coords = coords - coords[0]

            # Scale to reduce variation from hand size/distance
            scale = np.max(np.abs(coords))
            if scale > 0:
                coords = coords / scale

            values = coords.flatten()

            sample = pd.DataFrame(
                [values],
                columns=FEATURES
            )

            # Predict gesture and confidence
            probabilities = model.predict_proba(sample)[0]
            best_index = int(np.argmax(probabilities))
            predicted_label = str(model.classes_[best_index])
            confidence = float(probabilities[best_index])

            if confidence >= CONFIDENCE_THRESHOLD:
                recent_predictions.append(predicted_label)

                # Choose the most frequent recent prediction
                stable_label = Counter(
                    recent_predictions
                ).most_common(1)[0][0]

                display_text = LABEL_TO_TEXT.get(
                    stable_label,
                    stable_label
                )
            else:
                recent_predictions.clear()
                display_text = "Uncertain - try again"

        else:
            recent_predictions.clear()
            display_text = "Show a hand gesture"
            confidence = 0.0

        # -----------------------------
        # DISPLAY TEXT ON WEBCAM
        # -----------------------------
        cv2.rectangle(
            frame,
            (10, 10),
            (frame.shape[1] - 10, 110),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            frame,
            "SIGNOVA - ISL INTERPRETER",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "TEXT: " + display_text,
            (25, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.0%}",
            (25, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow("Signova - ISL Interpreter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("Interpreter stopped.")
