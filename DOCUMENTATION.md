# ASL Detector - Project Documentation

A clear, friendly guide to how this American Sign Language (ASL) detector works under the hood, the mathematical theory behind the feature engineering, how it was trained, and how to use it.

---

## 1. Overview

**ASL Detector** translates American Sign Language alphabet gestures into text in real time using a standard webcam. It recognizes **28 classes**:
- Letters **A through Z**
- **Space**
- **Del** (backspace / delete)

The project combines Google's **MediaPipe Hands** (for tracking 21 hand joints) with a regularized **Random Forest Classifier** from Scikit-Learn. It runs entirely on your CPU at 30+ FPS without needing a GPU.

---

## 2. Setup & Installation

### Requirements
- **Python**: Version **3.10 or 3.11** (MediaPipe's legacy `solutions.hands` works reliably up to Python 3.11).
- **Camera**: Any built-in or USB webcam.
- **OS**: Windows, macOS, or Linux.

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/Tanx-123/Sign-language-detector.git
cd Sign-language-detector

# 2. Create and activate a virtual environment
python -m venv .venv

# On Windows:
.\.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate

# 3. Install packages
pip install -r requirements.txt
```

Verify your installation:
```bash
python -c "import mediapipe as mp, cv2, sklearn; print('All libraries imported cleanly!')"
```

To run the live detector:
```bash
python app.py
```
*(Press **`q`** while in the OpenCV window to exit).*

---

## 3. Mathematical Formulations & Theory

MediaPipe tracks 21 hand joints $(x_i, y_i)$ in normalized image space $[0, 1]$. However, raw coordinates change whenever your hand moves around the screen or leans closer to the lens. 

To make detection robust across any camera distance or hand position, we convert raw landmarks into a **46-dimensional mathematical feature vector**:

```text
21 Raw Hand Landmarks: { (x_0, y_0), (x_1, y_1), ... , (x_20, y_20) }
                       │
                       ├── 1. Aspect-Preserved Scale & Translation Normalization (42 values)
                       │      x_norm = (x_i - min_x) / box_scale
                       │      y_norm = (y_i - min_y) / box_scale
                       │      where box_scale = max(width, height) + 1e-6
                       │
                       └── 2. Anatomical Palm-Scaled Knuckle Distances (4 values)
                              d(ThumbTip, Knuckle_j) = || P_4 - P_j ||_2 / palm_scale
                              where j in {13 (Ring), 9 (Middle), 17 (Pinky), 5 (Index)}
```

---

### 3.1. Scale & Translation Invariance Formulation

Given 21 detected joint coordinates $\{P_i = (x_i, y_i)\}_{i=0}^{20}$:

1. **Find Bounding Extrema**:
   $$x_{\min} = \min_{0 \le i \le 20}(x_i), \quad x_{\max} = \max_{0 \le i \le 20}(x_i)$$
   $$y_{\min} = \min_{0 \le i \le 20}(y_i), \quad y_{\max} = \max_{0 \le i \le 20}(y_i)$$

2. **Compute Aspect-Ratio Preserving Scale**:
   $$S_{\text{box}} = \max\Big(x_{\max} - x_{\min}, \; y_{\max} - y_{\min}\Big) + 10^{-6}$$

3. **Normalize Every Coordinate**:
   $$\hat{x}_i = \frac{x_i - x_{\min}}{S_{\text{box}}}, \quad \hat{y}_i = \frac{y_i - y_{\min}}{S_{\text{box}}} \quad \text{for } i = 0, 1, \dots, 20$$

* **Why this matters**:
  - **Translation Invariance**: Subtracting $x_{\min}, y_{\min}$ centers the hand at the origin $(0, 0)$. You can hold your hand in any corner of the webcam frame and the numbers remain identical.
  - **Scale Invariance**: Dividing by the larger dimension ($S_{\text{box}}$) maps all coordinates cleanly into $[0, 1]$ while preserving the hand's natural proportions. Moving your hand closer to or further from the camera does not alter the feature values.

---

### 3.2. Anatomical Palm Reference Metric ($S_{\text{palm}}$)

To measure finger distances without camera distortion, we need a rigid biological reference on the hand itself:

$$S_{\text{palm}} = \|P_9 - P_0\|_2 = \sqrt{(x_9 - x_0)^2 + (y_9 - y_0)^2} + 10^{-6}$$

* **Why the Wrist-to-Middle-MCP distance ($P_0 \rightarrow P_9$)?**
  The bone connecting your wrist to the base of your middle finger is the rigid center of the palm. Unlike fingers, it never flexes or curls. It acts as an invariant biological ruler.

---

### 3.3. Knuckle Geometry Disambiguation (M vs N vs T vs S)

In American Sign Language, closed-fist signs (like **M, N, T, S, A, E**) have all fingers curled. The only difference is where the thumb tip ($P_4$) rests relative to the four MCP knuckles ($P_5, P_9, P_{13}, P_{17}$):

$$d(4, j) = \frac{\|P_4 - P_j\|_2}{S_{\text{palm}}} = \frac{\sqrt{(x_4 - x_j)^2 + (y_4 - y_j)^2}}{S_{\text{palm}}}$$

- **$d(4, 13)$ (Thumb Tip to Ring MCP)**: Minimized in **M** (thumb tucked under 3 fingers).
- **$d(4, 9)$ (Thumb Tip to Middle MCP)**: Minimized in **N** (thumb tucked under 2 fingers).
- **$d(4, 17)$ (Thumb Tip to Pinky MCP)**: Measures thumb travel across the entire fist (helps separate **M** vs **S**).
- **$d(4, 5)$ (Thumb Tip to Index MCP)**: Minimized in **T** (thumb peeks between index and middle).

Together, $42 \text{ coordinates} + 4 \text{ inter-joint distances} = \mathbf{46 \text{ features}}$.

---

### 3.4. Mathematical Reflection Invariance (Data Augmentation)

To support both left and right hands without collecting a second dataset, we mirror training samples along the horizontal axis:

$$\hat{x}_i^{\text{flipped}} = \max(\hat{\mathbf{x}}) - \hat{x}_i, \quad \hat{y}_i^{\text{flipped}} = \hat{y}_i$$

* **Euclidean Invariance Property**:
  The distance between any two mirrored points $P_a$ and $P_b$ is strictly preserved:
  $$\|P_a^{\text{flip}} - P_b^{\text{flip}}\|_2 = \sqrt{\big(\max(\hat{\mathbf{x}}) - \hat{x}_a - (\max(\hat{\mathbf{x}}) - \hat{x}_b)\big)^2 + (\hat{y}_a - \hat{y}_b)^2} = \|P_a - P_b\|_2$$
  This proves mathematically that all 4 distance features are identical for left and right hands, requiring no manual recomputation during reflection.

---

### 3.5. Random Forest Decision Theory & Regularization

The classifier is an ensemble of $B = 100$ decision trees:

1. **Gini Impurity Splitting**:
   At each internal node $t$, the split threshold is chosen to maximize information gain (minimize impurity):
   $$I_G(t) = 1 - \sum_{c=1}^{C} p(c \mid t)^2$$
   where $p(c \mid t)$ is the proportion of class $c$ samples at node $t$.

2. **Ensemble Probability Estimation**:
   Individual tree outputs are averaged across all 100 trees to yield a calibrated class probability:
   $$P(y = c \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} P_b(y = c \mid \mathbf{x})$$
   $$\hat{y} = \arg\max_{c \in \{1, \dots, 28\}} P(y = c \mid \mathbf{x})$$

3. **Anti-Overfitting Regularization**:
   - `min_samples_leaf = 2`: Prevents trees from memorizing single noisy samples (forces each leaf to generalize across at least 2 hands).
   - `max_depth = 22`: Limits depth to prevent leaves from fitting to tiny coordinate jitter.
   - `max_samples = 0.8`: Trains each tree on a random 80% bootstrap slice, maximizing diversity across the forest and lowering variance.

---

### 3.6. Evaluation Metrics

1. **Multi-Class Cross-Entropy Log Loss**:
   Monitors prediction confidence convergence across tree count ($1 \rightarrow 100$):
   $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^{28} y_{i,c} \ln\big(p_{i,c}\big)$$

2. **Precision, Recall, and F1-Score**:
   $$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$
   $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 4. Model Training Pipeline (`model.ipynb`)

The model is trained on the [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (up to 1,000 images per class).

The training notebook follows a clean, leak-free workflow:

1. **Feature Extraction (`process_image`)**:
   Reads images from `asl_alphabet_train/` and computes the 46 features.
2. **Train-Test Split First**:
   Data is split **80% training / 20% testing** *before* any augmentation. This ensures the test set contains only genuine, unseen images with zero data leakage.
3. **Training-Only Augmentation**:
   Only the training set (`x_train`) is doubled with horizontal reflections, teaching the model both hands while keeping `x_test` untouched.
4. **Regularized Random Forest**:
   ```python
   model = RandomForestClassifier(
       n_estimators=1,
       max_depth=22,        # Prevents hyper-deep branches chasing coordinate noise
       min_samples_leaf=2,  # Every leaf must cover at least 2 distinct hand samples
       min_samples_split=5, # Requires sufficient data before creating a new branch
       max_samples=0.8,     # Trains each tree on an 80% bootstrap slice for diversity
       warm_start=True,
       random_state=42,
       n_jobs=-1
   )
   ```
5. **Progress & Evaluation**:
   Uses `warm_start=True` to log accuracy and log loss as trees grow from 1 to 100, saving:
   - `train_test_curves.png`: Training vs. testing accuracy and loss convergence curves.
   - `confusion_matrix.png`: Detailed 28×28 heatmap showing per-letter precision and recall.
   - `model.pickle` and `label_encoder.pickle`: Saved model artifacts for `app.py`.

---

## 5. Real-Time Detection (`app.py`)

The live application is optimized for speed, low CPU usage, and clean code:

- **Single-Pass Feature Extraction**: Bounding box coordinates (`min_x, max_x, min_y, max_y`) are calculated once and shared between feature normalization and visual box drawing.
- **Fast Distance Math**: Uses Python's built-in `math.hypot()` (implemented in C) instead of heavy NumPy array operations for distances in the video loop.
- **Instant Label Lookup**: Maps numeric model predictions to character strings with a direct array lookup (`label_classes[pred]`) instead of slow `inverse_transform` methods.
- **Safe Hardware Cleanup**: All webcam capture logic is wrapped in a `try ... finally` block, ensuring your webcam is cleanly released even if the program is closed abruptly.

---

## 6. Project Structure

```text
Sign-language-detector/
├── app.py                 # Real-time webcam detector with OpenCV
├── model.ipynb            # Dataset processing, feature extraction, training & evaluation
├── model.pickle           # Saved Random Forest model
├── label_encoder.pickle   # Label mapping for letters
├── train_test_curves.png  # Accuracy and loss graphs across 100 trees
├── confusion_matrix.png   # Full 28x28 confusion matrix plot
├── requirements.txt       # Dependencies with pinned versions
├── DOCUMENTATION.md       # Technical guide (this file)
└── README.md              # Project overview and quickstart guide
```

---

## 7. Performance

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Classes** | 28 | Letters A–Z, space, del |
| **Feature Dimensions** | 46 | 42 normalized coordinates + 4 finger distances |
| **Model** | Random Forest | 100 trees, regularized |
| **Test Accuracy** | ~98%+ | Evaluated on pure unaugmented 20% test partition |
| **Inference Speed** | ~20–30 ms / frame | Smooth 30–45 FPS on standard CPU |
| **Hardware** | Standard CPU | No GPU needed |

---

## 8. Common Issues & Easy Fixes

- **`AttributeError: module 'mediapipe' has no attribute 'solutions'`**:
  - *Cause*: Python 3.12+ or MediaPipe 1.0+ installed (Google moved legacy solutions).
  - *Fix*: Use Python 3.11 and install `mediapipe==0.10.14` from `requirements.txt`.
- **`[ERROR] Could not open webcam`**:
  - *Cause*: Another app (Zoom, Teams, browser) is using your webcam, or camera permissions are off.
  - *Fix*: Close other camera apps and ensure camera access is enabled in your OS settings.
- **Flickering or Inaccurate Detections**:
  - *Cause*: Backlighting, motion blur, or hand partly off-screen.
  - *Fix*: Make sure your hand is well-lit from the front and kept comfortably inside the camera frame.
