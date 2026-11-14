import cv2
import mediapipe as mp
import numpy as np
import joblib
import pickle

# ===================== MediaPipe Setup =====================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def main():
    # Load the trained model and label encoder
    print("[INFO] Loading model and label encoder...")
    try:
        with open('model.pickle', 'rb') as f:
            model_dict = pickle.load(f)
            model = model_dict['model']
        label_encoder = joblib.load('label_encoder.pickle')
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model files: {e}")
        return

    # Initialize MediaPipe Hands detector
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    # Initialize video capture with default camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam (index 0). Please check your camera connection.")
        return

    # Set frame properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] Starting real-time detection. Press 'q' in the OpenCV window to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame from camera.")
            break

        # Flip the frame horizontally for natural selfie-view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert the BGR frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks and connections using MediaPipe drawing utilities
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract landmark coordinates for this hand
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                min_x = min(x_coords)
                min_y = min(y_coords)

                # Normalize features relative to bounding box minimums (42 features)
                data_aux = []
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

                # Calculate bounding box coordinates
                x1 = max(0, int(min(x_coords) * w) - 10)
                y1 = max(0, int(min(y_coords) * h) - 10)
                x2 = min(w, int(max(x_coords) * w) + 10)
                y2 = min(h, int(max(y_coords) * h) + 10)

                # Predict gesture
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = label_encoder.inverse_transform([int(prediction[0])])[0]

                    # Draw bounding box and prediction text on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        str(predicted_character),
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA
                    )
                except Exception as e:
                    print(f"[ERROR] Prediction failed: {e}")

        # Display header overlay
        cv2.putText(
            frame,
            "ASL Detector (Press 'q' to quit)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Show frame in OpenCV window
        cv2.imshow("ASL Detector", frame)

        # Exit when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("[INFO] Application closed successfully.")

if __name__ == '__main__':
    main()
