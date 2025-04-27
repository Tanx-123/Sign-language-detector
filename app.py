import tkinter as tk
from tkinter import Frame, Label
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class ASLDetector:
    def __init__(self, master):
        self.master = master
        master.title("ASL Detector")
        master.geometry("800x600") # Set a default window size

        # Main frame
        main_frame = Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Video frame
        video_frame = Frame(main_frame)
        video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.video_label = Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Prediction frame
        prediction_frame = Frame(main_frame, height=50)
        prediction_frame.pack(side=tk.BOTTOM, fill=tk.X)
        prediction_frame.pack_propagate(False) # Prevent resizing

        self.prediction_text = tk.StringVar()
        self.prediction_text.set("Prediction: --")
        self.prediction_label = Label(prediction_frame, textvariable=self.prediction_text, font=("Helvetica", 16))
        self.prediction_label.pack(pady=10)

        # Load the trained model and label encoder
        model_dict = pickle.load(open('model.pickle', 'rb'))
        self.model = model_dict['model']
        self.label_encoder = joblib.load('label_encoder.pickle')

        # Initialize the hands module
        self.hands = mp_hands.Hands(
            static_image_mode=True,
            min_detection_confidence=0.3,
            max_num_hands=2
        )

        # Initialize the video capture
        self.cap = cv2.VideoCapture(0)

        # Start the video playback
        self.show_video()

    def show_video(self):
        # Capture a frame from the video
        ret, frame = self.cap.read()

        if ret:
            # Flip the frame horizontally for a more natural view
            frame = cv2.flip(frame, 1)

            # Process the frame for hand landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            # Draw hand landmarks and make predictions
            H, W, _ = frame.shape
            prediction_display = "No Hand Detected"

            if results.multi_hand_landmarks:
                prediction_display = "--" # Reset if hands are detected
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Prepare data for prediction (only process one hand for simplicity here)
                    data_aux = []
                    x_ = []
                    y_ = []
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    # Draw bounding box for the hand
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10 # Adjusted for better fit
                    y2 = int(max(y_) * H) + 10 # Adjusted for better fit
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Make prediction if data is valid
                    if len(data_aux) == 42: # Check if we have 21 landmarks * 2 coordinates
                        prediction = self.model.predict([np.asarray(data_aux)])
                        predicted_character = self.label_encoder.inverse_transform([int(prediction[0])])[0]
                        prediction_display = predicted_character
                    else:
                        prediction_display = "Processing..."

                    # Display prediction for the first detected hand only for now
                    break # Process only the first hand found

            # Update prediction label
            self.prediction_text.set(f"Prediction: {prediction_display}")

            # Convert the frame to RGB format and display it in the video label
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image)
            self.video_label.configure(image=photo)
            self.video_label.image = photo

            # Schedule the next frame update
            self.master.after(30, self.show_video)

root = tk.Tk()
app = ASLDetector(root)
root.mainloop()