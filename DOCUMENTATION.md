# ASL Detector - Complete Project Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Project Structure](#project-structure)
5. [Dataset Information](#dataset-information)
6. [Model Training Pipeline (`model.ipynb`)](#model-training-pipeline-modelipynb)
7. [Real-Time Inference (`app.py`)](#real-time-inference-apppy)
8. [Hand Landmark & Feature Extraction Details](#hand-landmark--feature-extraction-details)
9. [Usage Guide](#usage-guide)
10. [Technical Specifications & Performance](#technical-specifications--performance)
11. [Dependencies & Compatibility](#dependencies--compatibility)
12. [Troubleshooting](#troubleshooting)

---

## Project Overview

### What is ASL Detector?

**ASL Detector** is a real-time American Sign Language (ASL) recognition application that uses computer vision and machine learning to detect and classify ASL hand gestures directly from live webcam video. The application utilizes **MediaPipe Hands** (`mp.solutions.hands`) for robust 21-point hand landmark tracking and a trained **Random Forest Classifier** (`scikit-learn`) to recognize 28 ASL classes with low latency and native OpenCV rendering.

### Key Features

- **Real-time ASL Detection**: Recognizes **28 ASL classes** (Letters `A` through `Z`, plus `space` and `del`).
- **Multi-hand Support**: Detects and classifies up to **2 hands** simultaneously.
- **Position-Invariant Landmark Normalization**: Computes coordinates relative to each hand's bounding box minimums ($\Delta x_i = x_i - \min(x)$, $\Delta y_i = y_i - \min(y)$) to yield a 42-dimensional feature vector.
- **High Accuracy & Efficiency**: Delivers **~98.3% test accuracy** while running smoothly at 30+ FPS on CPU without requiring GPU acceleration.
- **Native OpenCV Display**: Direct window rendering (`cv2.imshow`) with bounding boxes, real-time landmark overlays, and prediction labels.

### Target Use Cases

- Learning and practicing American Sign Language alphabet gestures.
- Human-Computer Interaction (HCI) accessibility tools.
- Real-time gesture recognition prototyping and research.

---

## System Architecture

### High-Level Architecture Flow

```
Webcam Video Capture (cv2.VideoCapture)
    │
    ▼
Frame Preprocessing (Horizontal Mirror Flip + BGR ➔ RGB)
    │
    ▼
Hand Landmark Estimation (MediaPipe Hands: mp.solutions.hands)
    │
    ▼
Feature Extraction (21 Landmarks ➔ 42 Normalized Floats)
    │
    ▼
Gesture Classification (RandomForestClassifier .predict())
    │
    ▼
Label Decoding (LabelEncoder inverse_transform())
    │
    ▼
Frame Annotation (OpenCV Bounding Box, Skeleton, & Class Text)
    │
    ▼
Native UI Display (cv2.imshow with 'q' to exit)
```

### Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          Hardware                               │
│  Webcam (Default Index 0, 640×480 @ 30 FPS)                     │
└────────────────────────────────┬────────────────────────────────┘
                                 │ Frame Buffer
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      app.py (Main Loop)                         │
│  ├─ cv2.flip(frame, 1)                     [Natural mirror]     │
│  ├─ cv2.cvtColor(frame, COLOR_BGR2RGB)     [Color conversion]   │
│  ├─ mp_hands.Hands.process(frame_rgb)      [Landmark detection] │
│  ├─ extract_and_normalize_landmarks()       [42D vector]        │
│  ├─ model.predict([data_aux])              [RF classification]  │
│  ├─ label_encoder.inverse_transform()      [Decoded character]  │
│  └─ cv2.imshow("ASL Detector", frame)      [OpenCV display]     │
└────────────────────────────────┬────────────────────────────────┘
                                 │ Loads at Startup
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Trained Model Artifacts                     │
│  ├─ model.pickle         (RandomForestClassifier, 100 trees)    │
│  └─ label_encoder.pickle (LabelEncoder: 28 discrete classes)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation & Setup

### Prerequisites

- **Python**: Version **3.10** or **3.11** is recommended. *(Note: MediaPipe `0.10.x` solutions support Python up to 3.11. Python 3.12+ / 3.13 deprecates `mp.solutions.hands`).*
- **Camera**: Standard built-in or USB webcam.
- **Operating System**: Windows, Linux, or macOS.

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/Tanx-123/Sign-language-detector.git
cd Sign-language-detector
```

#### 2. Create and Activate a Virtual Environment

**Using Python 3.11 (`venv`):**
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

**Using `uv` (Fast installer):**
```bash
uv venv .venv --python 3.11
.\.venv\Scripts\activate
```

#### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

Verify that the packages are correctly installed:
```bash
python -c "import mediapipe as mp, cv2, sklearn; print('MediaPipe:', mp.__version__, '| Has solutions?', hasattr(mp, 'solutions'))"
```
*(Should output `Has solutions? True` with MediaPipe `0.10.14`).*

#### 4. Run the Real-Time Application

```bash
python app.py
```

Press **`q`** on the keyboard while focused on the OpenCV window to cleanly close the application.

---

## Project Structure

```
Sign-language-detector/
├── app.py                 # Real-time detection & inference with native OpenCV UI
├── model.ipynb            # Jupyter notebook for feature extraction and model training
├── model.pickle           # Trained Random Forest classifier (~87 MB)
├── label_encoder.pickle   # LabelEncoder mapping integers to character labels
├── requirements.txt       # Pinned library dependencies
├── .gitignore             # Git ignore configuration (virtual envs, pycache, data)
├── README.md              # Project overview and quickstart guide
└── DOCUMENTATION.md       # Comprehensive technical and pipeline documentation
```

### File Responsibilities

| File | Type | Description |
|---|---|---|
| `app.py` | Python Script | Captures webcam feed, runs landmark detection via MediaPipe, extracts 42 normalized features, classifies gestures, and displays annotations. |
| `model.ipynb` | Jupyter Notebook | Reads training images, performs horizontal flip data augmentation, runs MediaPipe feature extraction, trains Random Forest, and serializes the model. |
| `model.pickle` | Serialized Model | Contains `{'model': RandomForestClassifier}` trained on 42-dimensional landmark coordinate vectors. |
| `label_encoder.pickle` | Serialized Encoder | Maps discrete predictions (`0` to `27`) to ASL labels (`'A'` through `'Z'`, `'del'`, `'space'`). |
| `requirements.txt` | Configuration | Pinned library versions for reproducible environments. |

---

## Dataset Information

### Overview

The model is trained on the **ASL Alphabet Dataset**:
- **Source**: [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Image Resolution**: 200×200 pixels (RGB/BGR).
- **Target Classes (28 Classes)**:
  - Alphabet: `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `Z`
  - Control Gestures: `del`, `space`

### Dataset Sampling & Augmentation

In `model.ipynb`:
1. **Sampling**: Up to 1,000 images per class are sampled randomly to maintain balanced class representation across all 28 classes.
2. **Horizontal Flipping**: Each training image is processed in its original orientation and then horizontally flipped (`cv2.flip(img, 1)`).
3. **Robustness**: This mirrors hands left-to-right, effectively doubling the dataset and ensuring the model generalizes seamlessly to both left- and right-handed signers.

---

## Model Training Pipeline (`model.ipynb`)

The model training pipeline consists of 5 modular stages:

### Stage 1: MediaPipe Hands Initialization
```python
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
```
- `static_image_mode=True` forces MediaPipe to treat each training image as an independent detection target rather than relying on temporal tracking.

### Stage 2: Feature Extraction (`process_image`)
For each image:
1. MediaPipe detects up to 21 landmarks.
2. The coordinate minimums `min(x_)` and `min(y_)` across all 21 points are calculated.
3. Every landmark is normalized relative to the minimum:
   $$\Delta x_i = x_i - \min(x)$$
   $$\Delta y_i = y_i - \min(y)$$
4. A 42-dimensional feature vector is created: `[Δx_0, Δy_0, Δx_1, Δy_1, ..., Δx_20, Δy_20]`.

### Stage 3: Label Encoding
```python
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
joblib.dump(label_encoder, 'label_encoder.pickle')
```
Converts string categories (`'A'`, `'B'`, ...) into continuous integers (`0`, `1`, ...).

### Stage 4: Stratified Train-Test Split & Training
```python
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)
```
- **Stratification**: Guarantees identical class proportions in both training (80%) and testing (20%) sets.
- **Random Forest**: Constructs an ensemble of 100 decision trees with bootstrap aggregation (bagging) and random feature selection.

### Stage 5: Evaluation & Serialization
```python
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
# Achieves ~98.3% accuracy

with open('model.pickle', 'wb') as f:
    pickle.dump({'model': model}, f)
```

---

## Real-Time Inference (`app.py`)

### Execution Pipeline

```python
def main():
    # 1. Load model and label encoder
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)['model']
    label_encoder = joblib.load('label_encoder.pickle')

    # 2. Initialize MediaPipe in video streaming mode
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    # 3. Open webcam stream
    cap = cv2.VideoCapture(0)
    ...
```

### Key Differences in Real-Time Inference

1. **`static_image_mode=False`**:
   - In video mode, MediaPipe applies the palm detector only on the first frame or when tracking fails.
   - For consecutive frames, it uses a fast landmark tracking model, reducing latency to ~10–15ms per frame.
2. **Horizontal Mirroring (`cv2.flip(frame, 1)`)**:
   - Provides a natural selfie mirror experience for the user.
3. **Bounding Box Calculation**:
   - Converts normalized landmark extremes into frame pixel coordinates:
     $$x_1 = \max(0, \lfloor\min(x) \times W\rfloor - 10)$$
     $$y_1 = \max(0, \lfloor\min(y) \times H\rfloor - 10)$$
     $$x_2 = \min(W, \lceil\max(x) \times W\rceil + 10)$$
     $$y_2 = \min(H, \lceil\max(y) \times H\rceil + 10)$$
4. **Interactive Exit**:
   - Checks `cv2.waitKey(1) & 0xFF == ord('q')` to cleanly release the webcam and destroy all OpenCV windows.

---

## Hand Landmark & Feature Extraction Details

### 21 Hand Landmarks Map

```
          8 (INDEX_TIP)    12 (MIDDLE_TIP)   16 (RING_TIP)    20 (PINKY_TIP)
          │                     │                  │                │
          7 (INDEX_DIP)    11 (MIDDLE_DIP)   15 (RING_DIP)    19 (PINKY_DIP)
          │                     │                  │                │
          6 (INDEX_PIP)    10 (MIDDLE_PIP)   14 (RING_PIP)    18 (PINKY_PIP)
4 (THUMB_TIP)  │                │                  │                │
      │   5 (INDEX_MCP)     9 (MIDDLE_MCP)   13 (RING_MCP)    17 (PINKY_MCP)
3 (THUMB_IP)   │                │                  │                │
      │        └────────────────┴──────────────────┴────────────────┘
2 (THUMB_MCP)                           │
      │                                 │
1 (THUMB_CMC)                           │
      │                                 │
      └─────────────────────────────────0 (WRIST)
```

### Feature Normalization Math

Given 21 landmarks detected for a single hand:
$$\{(x_i, y_i)\}_{i=0}^{20}$$

1. Compute coordinate minimums:
   $$x_{\min} = \min_{0 \le i \le 20}(x_i), \quad y_{\min} = \min_{0 \le i \le 20}(y_i)$$
2. Form the 42-dimensional vector:
   $$\mathbf{v} = [x_0 - x_{\min}, \, y_0 - y_{\min}, \, x_1 - x_{\min}, \, y_1 - y_{\min}, \, \dots, \, x_{20} - x_{\min}, \, y_{20} - y_{\min}]$$

**Why this is effective:**
- **Translation Invariance**: The hand can be anywhere in the webcam's field of view without changing the feature vector.
- **Dimensional Simplicity**: 42 continuous floating-point features can be evaluated by a Random Forest in less than 2 milliseconds.

---

## Usage Guide

1. **Camera Position**: Sit approximately 0.5 to 1 meter away from the webcam with your hand comfortably in view.
2. **Lighting**: Ensure even lighting on the palm and fingers. Harsh backlighting or deep shadows can impair landmark tracking.
3. **Multiple Hands**: Up to 2 hands are processed simultaneously in the same frame.
4. **Termination**: Press the **`q`** key while focused on the video window to quit.

---

## Technical Specifications & Performance

| Parameter | Specification | Notes |
|---|---|---|
| **Input Resolution** | 640 × 480 @ 30 FPS | Standard webcam format |
| **Model Type** | Random Forest Classifier | `n_estimators=100`, scikit-learn |
| **Model Size** | ~86.9 MB | Serialized in `model.pickle` |
| **Feature Dimensions** | 42 | 21 landmarks × 2 coordinates ($\Delta x, \Delta y$) |
| **Test Accuracy** | **~98.30%** | Evaluated on stratified 20% test partition |
| **Inference Latency** | ~20–30 ms per frame | Real-time ~30–45 FPS on standard CPU |
| **Hardware Required** | Standard x86_64 CPU | GPU not required |

---

## Dependencies & Compatibility

### Pinned Dependencies (`requirements.txt`)

```text
opencv-python==4.11.0.86
mediapipe==0.10.14
numpy==1.26.4
scikit-learn==1.4.1.post1
joblib==1.6.0
```

### Compatibility Notes

- **Python Version**: Recommended on **Python 3.10 / 3.11**.
- **MediaPipe Version**: Must use `mediapipe<=0.10.14`. Versions $\ge$ 1.0.0 remove the `mp.solutions.hands` namespace in favor of the Tasks API.
- **Scikit-Learn Version**: Pinned to `1.4.1.post1` to prevent unpickling warnings and serialization incompatibilities with `model.pickle`.

---

## Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | MediaPipe version is $\ge$ 1.0.0 or running under Python 3.13 where legacy solutions are unavailable. | Use a Python 3.11 environment and install `mediapipe==0.10.14`. |
| `InconsistentVersionWarning` on unpickling | The installed `scikit-learn` version does not match the version used during training. | Install `scikit-learn==1.4.1.post1`. |
| `[ERROR] Could not open webcam` | Webcam index is incorrect, camera is in use by another application, or camera permissions are missing. | Verify webcam permissions and verify device index in `cv2.VideoCapture(0)`. |
| Predictions are inaccurate or jittery | Poor lighting, hand partially cut off, or motion blur. | Improve ambient lighting and hold gesture steady inside the frame. |

---
