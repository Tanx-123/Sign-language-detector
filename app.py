from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from contextlib import asynccontextmanager
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pickle
import threading

# ===================== Configuration Constants =====================
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 30
MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.3
TRACKING_CONFIDENCE = 0.3
JPEG_QUALITY = 80
BBOX_PADDING = 10
STATUS_TEXT_POSITION = (10, 30)
STATUS_TEXT_FONT_SCALE = 0.7
PREDICTION_TEXT_FONT_SCALE = 1.3
BBOX_COLOR = (0, 255, 0)  # Green in BGR
BBOX_THICKNESS = 2
TEXT_THICKNESS = 3
TEXT_OFFSET_Y = 10

# ===================== Initialize FastAPI App =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown events"""
    # Startup
    initialize_camera()
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    
    yield
    
    # Shutdown
    if camera_capture:
        camera_capture.release()
    if hand_detector:
        hand_detector.close()

app = FastAPI(title="ASL Detector", lifespan=lifespan)

# ===================== Load ML Models =====================
def load_model_files():
    """Load the trained model and label encoder"""
    try:
        with open('model.pickle', 'rb') as model_file:
            model_dict = pickle.load(model_file)
            loaded_model = model_dict['model']
        
        loaded_label_encoder = joblib.load('label_encoder.pickle')
        return loaded_model, loaded_label_encoder
    except FileNotFoundError:
        print("Warning: Model files not found. Predictions will not work.")
        return None, None

model, label_encoder = load_model_files()

# ===================== MediaPipe Setup =====================
mp_hands = mp.solutions.hands
hand_detector_drawing = mp.solutions.drawing_utils
hand_detector_styles = mp.solutions.drawing_styles

camera_capture = None
hand_detector = None
frame_lock = threading.Lock()
latest_processed_frame = None
latest_detected_gestures = []

def initialize_camera():
    """Initialize camera and hand detector at application startup"""
    global camera_capture, hand_detector
    
    camera_capture = cv2.VideoCapture(CAMERA_INDEX)
    
    # Configure camera properties
    camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera_capture.set(cv2.CAP_PROP_FPS, FRAME_RATE)
    
    # Initialize MediaPipe hand detector
    hand_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE
    )

def extract_and_normalize_landmarks(hand_landmarks):
    """Extract landmark coordinates and normalize them relative to hand bounds"""
    x_coordinates = [landmark.x for landmark in hand_landmarks.landmark]
    y_coordinates = [landmark.y for landmark in hand_landmarks.landmark]
    
    min_x, min_y = min(x_coordinates), min(y_coordinates)
    
    # Normalize: subtract minimum to make hand position relative
    normalized_features = []
    for landmark in hand_landmarks.landmark:
        normalized_features.append(landmark.x - min_x)
        normalized_features.append(landmark.y - min_y)
    
    return normalized_features, x_coordinates, y_coordinates

def predict_gesture(hand_features):
    """Predict ASL gesture from normalized hand features"""
    if not model or not label_encoder:
        return None
    try:
        prediction = model.predict([np.asarray(hand_features)])
        predicted_character = label_encoder.inverse_transform([int(prediction[0])])[0]
        return predicted_character
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


def draw_hand_annotation(frame, gesture_label, x_coordinates, y_coordinates):
    """Draw bounding box and gesture label on frame"""
    frame_height, frame_width = frame.shape[:2]
    # Calculate bounding box in pixel coordinates
    box_left = int(min(x_coordinates) * frame_width) - BBOX_PADDING
    box_top = int(min(y_coordinates) * frame_height) - BBOX_PADDING
    box_right = int(max(x_coordinates) * frame_width) + BBOX_PADDING
    box_bottom = int(max(y_coordinates) * frame_height) + BBOX_PADDING
    
    # Draw bounding box
    cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), 
                  BBOX_COLOR, BBOX_THICKNESS)
    
    # Draw gesture label
    cv2.putText(frame, gesture_label, (box_left, box_top - TEXT_OFFSET_Y),
                cv2.FONT_HERSHEY_SIMPLEX, PREDICTION_TEXT_FONT_SCALE, 
                BBOX_COLOR, TEXT_THICKNESS)


def detect_hand_gesture(frame):
    """Detect hand gestures and return annotated frame with predictions"""
    global latest_detected_gestures
    
    # Convert BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_results = hand_detector.process(frame_rgb)
    
    detected_gestures = []
    
    if detection_results.multi_hand_landmarks:
        for hand_landmarks in detection_results.multi_hand_landmarks:
            # Draw hand skeleton
            hand_detector_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                hand_detector_styles.get_default_hand_landmarks_style(),
                hand_detector_styles.get_default_hand_connections_style()
            )
            
            # Extract and normalize hand features
            normalized_features, x_coords, y_coords = extract_and_normalize_landmarks(hand_landmarks)
            
            # Predict gesture
            predicted_character = predict_gesture(normalized_features)
            
            if predicted_character:
                draw_hand_annotation(frame, predicted_character, x_coords, y_coords)
                detected_gestures.append(predicted_character)
    
    latest_detected_gestures = detected_gestures
    return frame, detected_gestures

def capture_frames():
    """Continuously capture and process frames in a background thread"""
    global latest_processed_frame
    while True:
        try:
            success, frame = camera_capture.read()
            if not success:
                break
            
            # Mirror frame for natural UI interaction
            frame = cv2.flip(frame, 1)
            # Detect hand gestures
            frame, _ = detect_hand_gesture(frame)
            
            # Add status text
            cv2.putText(frame, "ASL Detector - Live Detection", STATUS_TEXT_POSITION,
                       cv2.FONT_HERSHEY_SIMPLEX, STATUS_TEXT_FONT_SCALE, BBOX_COLOR, BBOX_THICKNESS)
            
            # Store frame safely with thread lock
            with frame_lock:
                latest_processed_frame = frame.copy()
        
        except Exception as e:
            print(f"Frame capture error: {e}")
            break

def generate_frames():
    """Generate video frames as MJPEG stream for browser display"""
    while True:
        with frame_lock:
            if latest_processed_frame is None:
                # Create black placeholder frame if no frame yet
                frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            else:
                frame = latest_processed_frame.copy()
        # Encode frame as JPEG
        success, jpeg_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not success:
            continue
        
        frame_bytes = jpeg_buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'+ frame_bytes + b'\r\n')

@app.get("/")
async def index():
    """Serve the main HTML page"""
    return FileResponse("index.html")

@app.get("/video_feed")
async def video_feed():
    """Stream video frames as MJPEG"""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/latest_predictions")
async def get_latest_predictions():
    """Get the latest detected hand gestures"""
    return {"predictions": latest_detected_gestures}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)