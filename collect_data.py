
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# ==========================================
# SIGNOVA: ISL GESTURE DATA COLLECTION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
CSV_FILE = os.path.join(DATA_DIR, "isl_landmarks.csv")

# Map keyboard keys to internal labels and displayed words
GESTURES = {
    ord("1"): ("SIGN_1", "OKAY"),
    ord("2"): ("SIGN_2", "STOP"),
    ord("3"): ("SIGN_3", "THANKS"),
    ord("4"): ("SIGN_4", "PEACE"),
    ord("5"): ("SIGN_5", "GOOD"),
}

FEATURES = [f"f{i}" for i in range(63)]
COLUMNS = FEATURES + ["label"]

# ==========================================
# MEDIAPIPE SETUP
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ==========================================
# OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    hands.close()
    raise RuntimeError(
        "Could not open webcam. Close other camera applications "
        "and check your camera connection."
    )

# Create CSV with headers if it doesn't exist or is empty
if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
    pd.DataFrame(columns=COLUMNS).to_csv(CSV_FILE, index=False)

selected_label = None
selected_word = "None"
saved_count = 0

# ==========================================
# MAIN LOOP
# ==========================================

try:
    while True:
        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        coords = None

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            # Draw hand landmarks on the webcam
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract 21 landmarks, each with x, y, z
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand.landmark],
                dtype=np.float32
            )

            # Normalize relative to the wrist
            coords = coords - coords[0]

            # Scale coordinates
            scale = np.max(np.abs(coords))
            if scale > 0:
                coords = coords / scale

        # Display selected word
        cv2.putText(
            frame,
            f"Selected: {selected_word}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "1: OKAY | 2: STOP | 3: THANKS",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "4: PEACE | 5: GOOD",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "SPACE: SAVE SAMPLE | Q: QUIT",
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        if coords is None:
            cv2.putText(
                frame,
                "No hand detected",
                (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        cv2.imshow("Signova - Gesture Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        # Select a gesture
        if key in GESTURES:
            selected_label, selected_word = GESTURES[key]
            print(f"Selected: {selected_label} -> {selected_word}")

        # Save the current hand coordinates
        elif key == ord(" "):
            if selected_label is None:
                print("Select a gesture first using keys 1-5.")
            elif coords is None:
                print("No hand detected. Show your hand and try again.")
            else:
                row = coords.flatten().tolist()
                row.append(selected_label)

                pd.DataFrame(
                    [row], columns=COLUMNS
                ).to_csv(
                    CSV_FILE,
                    mode="a",
                    header=False,
                    index=False
                )

                saved_count += 1
                print(
                    f"Saved sample {saved_count}: "
                    f"{selected_word} ({selected_label})"
                )

        # Quit
        elif key == ord("q"):
            break

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("Data collection stopped.")
    print(f"New samples saved this session: {saved_count}")
    print(f"Dataset file: {CSV_FILE}")
