# ASL Detector - Real-time American Sign Language Recognition

A real-time American Sign Language (ASL) recognition application that uses computer vision and machine learning to detect and classify 29 ASL characters (A-Z, space, delete, nothing) from webcam input. The application features a FastAPI backend with live MJPEG video streaming and a web-based interface.

## Key Features

- **Real-time ASL Detection**: Recognizes 29 American Sign Language characters with high accuracy
- **Multi-hand Support**: Detects and classifies up to 2 hands simultaneously
- **Web-Based Interface**: FastAPI backend with live MJPEG video streaming accessible via any modern browser
- **MediaPipe Integration**: Uses MediaPipe for robust hand landmark detection
- **Random Forest Classifier**: Trained ML model for accurate sign recognition
- **Thread-Safe Processing**: Background thread architecture for smooth performance
- **Easy Deployment**: Simple HTTP endpoints for video streaming and predictions

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Tanx-123/Sign-language-detector.git
    cd Sign-language-detector
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Start the application:**
   ```bash
   python app.py
   ```
   The FastAPI server will start and listen on `http://localhost:8000`

2. **Open the web interface:**
   - Open your browser and navigate to `http://localhost:8000`
   - The application will display a live video feed from your webcam with real-time ASL detection

3. **How to Use:**
   - Position your hand(s) within the camera frame
   - The application will detect hand landmarks using MediaPipe
   - Detected ASL signs will be displayed with bounding boxes and labels
   - Your predictions appear in real-time as you perform different signs

## API Endpoints

The application provides the following REST endpoints:

- `GET /` - Serves the main HTML interface
- `GET /video_feed` - Streams live MJPEG video with hand detection annotations
- `GET /latest_predictions` - Returns the latest detected hand gestures as JSON

## Sample Video

https://github.com/Tanx-123/Sign-language-detector/assets/75534311/a66155ce-d1d8-40a7-bf14-1577384785f4

## Documentation

For detailed information about the project architecture, model training, dataset structure, and technical implementation, see [DOCUMENTATION.md](DOCUMENTATION.md).


