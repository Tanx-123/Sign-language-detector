import math
import pickle
import cv2
import joblib
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def main():
    # Load trained model and class labels
    try:
        with open('model.pickle', 'rb') as f:
            model = pickle.load(f)['model']
        label_classes = joblib.load('label_encoder.pickle').classes_
        print("[INFO] Model and labels loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model files: {e}")
        return

    # Initialize MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam (index 0).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[INFO] Real-time detection active. Press 'q' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror frame for natural selfie view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Process frame with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Extract raw coordinates
                    raw_x = [lm.x for lm in hand_landmarks.landmark]
                    raw_y = [lm.y for lm in hand_landmarks.landmark]

                    # Scale and translation normalization
                    min_x, max_x = min(raw_x), max(raw_x)
                    min_y, max_y = min(raw_y), max(raw_y)
                    box_scale = max(max_x - min_x, max_y - min_y) + 1e-6

                    features = []
                    for x, y in zip(raw_x, raw_y):
                        features.extend([(x - min_x) / box_scale, (y - min_y) / box_scale])

                    # Add palm-scale normalized finger distances
                    scale = math.hypot(raw_x[9] - raw_x[0], raw_y[9] - raw_y[0]) + 1e-6
                    features.extend([
                        math.hypot(raw_x[4] - raw_x[13], raw_y[4] - raw_y[13]) / scale,
                        math.hypot(raw_x[4] - raw_x[9],  raw_y[4] - raw_y[9])  / scale,
                        math.hypot(raw_x[4] - raw_x[17], raw_y[4] - raw_y[17]) / scale,
                        math.hypot(raw_x[4] - raw_x[5],  raw_y[4] - raw_y[5])  / scale
                    ])

                    # Calculate bounding box coordinates
                    x1, y1 = max(0, int(min_x * w) - 10), max(0, int(min_y * h) - 10)
                    x2, y2 = min(w, int(max_x * w) + 10), min(h, int(max_y * h) + 10)

                    # Predict gesture and draw label
                    try:
                        pred = model.predict(np.array([features], dtype=np.float32))[0]
                        predicted_char = label_classes[pred]

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            str(predicted_char),
                            (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.3,
                            (0, 255, 0),
                            3,
                            cv2.LINE_AA
                        )
                    except Exception as e:
                        print(f"[ERROR] Prediction failed: {e}")

            cv2.putText(frame, "ASL Detector (Press 'q' to quit)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("ASL Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release resources on exit
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("[INFO] Application closed.")


if __name__ == '__main__':
    main()
