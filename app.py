import tkinter as tk
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

        # Create a label to display the video feed
        self.label = tk.Label(master)
        self.label.pack()

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
            data_aux = []
            x_ = []
            y_ = []
            H, W, _ = frame.shape
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Extract hand features
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

                    # Draw bounding box for each hand
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10

                    # Make predictions using the trained model
                    prediction = self.model.predict([np.asarray(data_aux)])
                    predicted_character = self.label_encoder.inverse_transform([int(prediction[0])])[0]
                    # Display the predicted character on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                    cv2.putText(frame, predicted_character, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                    # Reset data_aux, x_, and y_ for the next hand
                    data_aux = []
                    x_ = []
                    y_ = []

            # Convert the frame to RGB format and display it in the label
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image)
            self.label.configure(image=photo)
            self.label.image = photo

            # Schedule the next frame update
            self.master.after(30, self.show_video)

root = tk.Tk()
app = ASLDetector(root)
root.mainloop()