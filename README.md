# Real-Time Driver Drowsiness Detection System

## Problem Statement

Driver drowsiness is one of the leading causes of road accidents worldwide.
Fatigued drivers often fail to notice early warning signs of drowsiness,
making automated detection a critical safety tool.

## Solution Approach

This system uses a standard webcam and classical computer vision techniques
(Haar Cascade classifiers via OpenCV) to detect whether a driver's eyes are
closed over several consecutive frames. When the eye-closure persists beyond
a configurable threshold, both a visual banner and an audible alarm are
triggered in real time.

---

## How It Works (Step by Step)

1. **Capture** — OpenCV reads frames from the default webcam at 640 × 480.
2. **Pre-process** — Each frame is converted to grayscale and histogram-equalised
   to handle varying lighting conditions.
3. **Face detection** — A Haar Cascade (`haarcascade_frontalface_default.xml`)
   locates the driver's face in the frame.
4. **Eye detection** — A second Haar Cascade (`haarcascade_eye.xml`) searches the
   upper 60 % of the detected face region for eyes.
5. **Drowsiness logic** — If no eyes are found inside the face region, an internal
   counter is incremented. If at least one eye is detected the counter resets.
6. **Alert** — Once the counter reaches the threshold (default: **20 frames**),
   a blinking red banner is overlaid on the video feed and a short alarm beep is
   played through the speakers.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Video / CV | OpenCV 4 |
| Numerical ops | NumPy |
| Audio alert | Pygame (mixer) |
| Model files | OpenCV Haar Cascades (XML) |

---

## Project Structure

```
drowsiness-detection-system/
├── models/
│   ├── haarcascade_frontalface_default.xml   # Pre-trained face detector
│   └── haarcascade_eye.xml                   # Pre-trained eye detector
├── src/
│   ├── __init__.py
│   ├── detection.py   # DrowsinessDetector class — all CV logic
│   ├── alert.py       # Visual warning overlay + audio alarm
│   └── utils.py       # Cascade loader, draw helpers
├── app.py             # Entry point — main loop
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the system

```bash
python app.py
```

A window titled **"Driver Drowsiness Detection"** will open showing your webcam feed.

### 3. Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the application |
| `R` | Reset the closed-eye frame counter |

---

## Configuration

Open `app.py` and adjust these constants at the top of the file:

| Constant | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Webcam device index (try `1` or `2` for external cameras) |
| `FRAME_WIDTH` | `640` | Capture width in pixels |
| `FRAME_HEIGHT` | `480` | Capture height in pixels |
| `DROWSY_THRESHOLD` | `20` | Consecutive closed-eye frames before alert fires |

To make detection more sensitive (alert faster), lower `DROWSY_THRESHOLD`.
To reduce false alarms in poor lighting, raise it.

---

## Future Improvements

- **Eye Aspect Ratio (EAR)** — Use facial landmark detection (dlib / MediaPipe) to
  compute EAR for more precise eye-state classification, especially under partial
  occlusion.
- **Deep learning classifier** — Replace the Haar Cascade eye detector with a small
  CNN trained on the MRL Eye Dataset for higher accuracy across diverse lighting.
- **Head-pose estimation** — Detect nodding or head-drop as an additional fatigue signal.
- **Yawn detection** — Detect open-mouth yawns as a secondary drowsiness indicator.
- **Data logging** — Record drowsiness events with timestamps for fleet management
  dashboards.
- **Mobile deployment** — Package the model for on-device inference on a dashcam
  running Android (TFLite) or iOS (CoreML).
