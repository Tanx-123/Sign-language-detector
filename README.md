# ASL Detector - Real-time American Sign Language Recognition

A real-time American Sign Language (ASL) recognition application that uses computer vision and machine learning to detect and classify 28 ASL characters (A–Z, space, del) from live webcam input. The application uses MediaPipe Hands for robust hand landmark tracking, extracts normalized geometric features, and classifies gestures using a trained Random Forest classifier with native real-time OpenCV display.

## Key Features

- **Real-time ASL Detection**: Accurately recognizes 28 American Sign Language characters (A–Z, space, del).
- **Multi-hand Support**: Detects and classifies up to 2 hands simultaneously in real time.
- **MediaPipe Landmark Tracking**: Tracks 21 3D hand keypoints per hand for precise, jitter-free joint tracking.
- **Position-Invariant Feature Extraction**: Normalizes landmark coordinates relative to bounding box minimums (42 features per hand) so signs are recognized anywhere in the camera view.
- **Data-Leak-Free Training Pipeline**: Features are split into train and test sets *before* data augmentation, ensuring evaluation metrics are genuine and reliable.
- **Warm-Start Training & Evaluation Visualizations**: Tracks training progress per decision tree, generating convergence curves for accuracy, multi-class log loss (`train_test_curves.png`), and a full confusion matrix (`confusion_matrix.png`).
- **Random Forest Classifier**: High-performance, lightweight ensemble classifier (~98%+ test accuracy) enabling low-latency real-time inference without needing a GPU.
- **Native OpenCV UI**: Direct low-latency camera capture and interactive display window with bounding box overlays and prediction labels.

## Prerequisites

- **Python**: Python 3.10 or 3.11 is recommended (MediaPipe `0.10.x` solutions support Python up to 3.11).
- **Webcam**: A built-in or USB webcam.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Tanx-123/Sign-language-detector.git
   cd Sign-language-detector
   ```

2. **Create and activate a virtual environment:**
   Using standard `venv` (Python 3.10/3.11):
   ```bash
   python -m venv .venv
   # Windows (Command Prompt / PowerShell)
   .\.venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

   *Alternatively, using `uv`:*
   ```bash
   uv venv .venv --python 3.11
   .\.venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the detector:**
   ```bash
   python app.py
   ```

2. **How to interact:**
   - Position your hand(s) in front of the webcam.
   - The green skeleton landmarks, bounding box, and predicted ASL character will overlay in real-time.
   - Press **`q`** on your keyboard while focused on the OpenCV window to exit.

3. **Train or evaluate the model:**
   - Open [`model.ipynb`](model.ipynb) in Jupyter Notebook or VS Code.
   - Step through the cells to extract features, train the Random Forest with warm-start per-tree logging, plot accuracy/loss curves, and inspect the per-class confusion matrix.

## Project Structure

```text
Sign-language-detector/
├── app.py                 # Real-time detection & OpenCV inference script
├── model.ipynb            # Dataset processing, feature extraction, training & evaluation pipeline
├── model.pickle           # Serialized Random Forest classifier
├── label_encoder.pickle   # Serialized LabelEncoder mapping integer predictions to letters
├── train_test_curves.png  # Generated training/testing loss and accuracy convergence plot
├── confusion_matrix.png   # Generated 28x28 confusion matrix heatmap
├── requirements.txt       # Pinned library dependencies
├── DOCUMENTATION.md       # Detailed technical documentation & pipeline overview
└── README.md              # Project overview & quickstart guide
```

## How It Works

```
 Webcam Frame (640x480 BGR)
         │
         ▼
 Horizontal Flip (Selfie Mirror) & BGR ➔ RGB
         │
         ▼
 MediaPipe Hands (21 Landmarks per hand)
         │
         ▼
 Feature Normalization (Δx_i = x_i - min_x, Δy_i = y_i - min_y ➔ 42 features)
         │
         ▼
 Random Forest Model (.predict())
         │
         ▼
 Label Decoding (e.g., 0 ➔ 'A') & OpenCV Rendering (Bounding Box + Text)
```

## Sample Video

https://github.com/Tanx-123/Sign-language-detector/assets/75534311/a66155ce-d1d8-40a7-bf14-1577384785f4

## Documentation

For a comprehensive technical breakdown including model training, architecture diagrams, mathematical formulations, and detailed viva Q&A, refer to [DOCUMENTATION.md](DOCUMENTATION.md).
