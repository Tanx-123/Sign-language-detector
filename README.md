# ASL Detector - Real-time American Sign Language Recognition

A lightweight, real-time sign language recognition tool built with Python, OpenCV, MediaPipe, and Scikit-Learn. It detects hand gestures from your webcam and translates them into 28 American Sign Language (ASL) characters (letters A–Z, space, and delete) at 30+ FPS right on your CPU — no expensive GPU required.

---

## What It Does

- **Recognizes 28 Signs**: Detects all 26 alphabet letters (`A` to `Z`) plus `space` and `del`.
- **Works at Any Distance**: Landmarks are normalized for both position and distance so your hand is recognized whether you're close to the camera or sitting back.
- **Up to 2 Hands**: Tracks and labels up to two hands at once in live video.
- **Fast and Lightweight**: Runs on regular CPU hardware (~20–30 ms per frame) using a regularized Random Forest model (~98%+ accuracy).
- **Clean OpenCV Window**: Overlays a clean bounding box, hand skeleton lines, and the detected letter directly on your webcam feed.

---

## Quickstart

### 1. Prerequisites
- **Python 3.10 or 3.11** (MediaPipe 0.10.x works best on Python 3.11).
- A standard webcam (built-in or USB).

### 2. Setup

Clone the repository and set up a virtual environment:

```bash
# Clone the repo
git clone https://github.com/Tanx-123/Sign-language-detector.git
cd Sign-language-detector

# Create a virtual environment
python -m venv .venv

# Activate it
# On Windows:
.\.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the App

Start the live webcam detector:

```bash
python app.py
```

- Hold your hand up to the camera to see live predictions.
- Press **`q`** on your keyboard to close the window.

---

## Training Your Own Model

If you want to retrain the model or explore the training data:

1. Download the [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and place the `asl_alphabet_train` folder in the project root.
2. Open [`model.ipynb`](model.ipynb) in VS Code or Jupyter Notebook.
3. Run all cells:
   - Extracts 46 normalized landmark features per hand.
   - Splits train and test data before augmenting with horizontal flips (avoids data leakage).
   - Trains a regularized Random Forest with per-tree tracking.
   - Generates [`train_test_curves.png`](train_test_curves.png) and a detailed [`confusion_matrix.png`](confusion_matrix.png).
   - Saves the trained model to `model.pickle` and label mapping to `label_encoder.pickle`.

---

## How It Works

```text
Webcam Frame (640x480)
       │
       ▼
Mirror Image & Color Convert (BGR ➔ RGB)
       │
       ▼
MediaPipe Hands (21 Landmark Coordinates)
       │
       ▼
46-Dimensional Feature Vector
  ├── 42 coordinates (normalized for position & hand scale)
  └── 4 thumb-to-knuckle distances (normalized by palm size)
       │
       ▼
Random Forest Classifier ➔ Predicted Letter Displayed on Screen
```

---

## Project Structure

```text
Sign-language-detector/
├── app.py                 # Real-time webcam detector with OpenCV
├── model.ipynb            # Feature extraction, training, and evaluation notebook
├── model.pickle           # Saved Random Forest classifier
├── label_encoder.pickle   # Label mapping (0-27 to letters A-Z, del, space)
├── train_test_curves.png  # Accuracy and loss graphs across 100 trees
├── confusion_matrix.png   # Full 28x28 confusion matrix plot
├── requirements.txt       # Dependencies with pinned versions
├── DOCUMENTATION.md       # In-depth technical documentation
└── README.md              # Project overview and setup guide
```

---

## Tips for Best Detection

- **Lighting**: Good, even lighting on your hand helps MediaPipe track your knuckles reliably.
- **Background**: A clean or uncluttered background makes hand edge detection sharper.
- **Hand Angle**: Keep your palm facing generally toward the camera, matching standard ASL chart poses.

---

## More Details

For complete technical details, mathematical formulas, training metrics, and troubleshooting steps, check out [DOCUMENTATION.md](DOCUMENTATION.md).
