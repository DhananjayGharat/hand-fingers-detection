# Hand & Fingers Detection using Python

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.x-0097A7?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A beginner-friendly, real-time Computer Vision application that detects human hands, tracks 21 skeletal landmarks, classifies handedness, and counts raised fingers live through your webcam.

---

## Overview

This project uses a webcam to detect human hands in real time and track 21 3D hand landmarks using **MediaPipe** and **OpenCV**. It provides an interactive computer vision experience with an on-screen Heads-Up Display (HUD) showing the number of detected hands, handedness classification (Left vs. Right hand), individual finger states, a live finger counter, and real-time frames-per-second (FPS) metrics.

---

## Features

- **Real-time webcam hand detection**: Streams and processes frames directly from your default camera.
- **21 hand landmark detection**: Detects key joints, knuckles, and fingertips in 3D normalized coordinates.
- **Hand skeleton visualization**: Renders custom-styled landmark nodes and skeletal connections.
- **Left/right hand identification**: Automatically identifies and labels whether a hand is Left or Right in mirrored selfie view.
- **Finger counting**: Accurately counts how many fingers are extended (0 to 5 per hand) using landmark coordinate geometry.
- **Support for detecting multiple hands**: Detects and tracks up to 2 hands simultaneously.
- **Real-time FPS display**: Monitored and rendered live on the HUD for performance tracking.
- **Keyboard control to exit the application**: Press **`Q`** at any time to smoothly exit and release camera hardware.

---

## Technologies

- **Python**: Primary programming language for application logic.
- **OpenCV (`opencv-python`)**: Video capture, frame flipping, color space conversions, and HUD overlay rendering.
- **MediaPipe**: High-fidelity machine learning pipeline for real-time multi-hand landmark detection.
- **NumPy**: Matrix and multi-dimensional image array processing used internally by OpenCV and MediaPipe.

---

## How It Works

The processing pipeline executes frame-by-frame as follows:

```text
Webcam
  ↓
OpenCV frame capture
  ↓
MediaPipe Hands
  ↓
21 hand landmarks
  ↓
Finger-state detection
  ↓
Finger counting
  ↓
Real-time visualization
```

### Landmark Anatomy & Coordinates

MediaPipe identifies 21 normalized landmarks per hand \((x, y, z) \in [0.0, 1.0]\):

```text
       8   12  16  20        Fingertips:
       |   |   |   |         - Thumb: 4
   4   7  11  15  19         - Index: 8
   |   |   |   |   |         - Middle: 12
   3   6  10  14  18         - Ring: 16
   |   |   |   |   |         - Pinky: 20
   2---5---9--13--17
    \              /         Knuckles / PIP Joints:
     1            /          - Thumb IP: 3
      \          /           - Index PIP: 6
       \        /            - Middle PIP: 10
        \      /             - Ring PIP: 14
           0 (Wrist)         - Pinky PIP: 18
```

1. **Coordinate System**: In image coordinates, `(0,0)` is at the **top-left corner**. The Y-axis increases downwards, and the X-axis increases to the right.
2. **Four Extended Fingers (Index, Middle, Ring, Pinky)**: A finger is considered raised when its fingertip has a smaller Y-coordinate than its corresponding PIP joint (i.e. `landmark[tip].y < landmark[pip].y`).
3. **The Thumb**: Because the thumb articulates sideways, horizontal position is evaluated:
   - For a **Right Hand** (mirror view): Thumb is extended when `landmark[4].x < landmark[3].x`.
   - For a **Left Hand** (mirror view): Thumb is extended when `landmark[4].x > landmark[3].x`.

---

## Installation

Follow these steps to set up the project locally:

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/hand-fingers-detection.git
cd hand-fingers-detection
```

### 2. Create a Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the main Python script:

```bash
python main.py
```

### How to Use
1. Hold your hand(s) up in front of your webcam, facing the camera.
2. The HUD overlay in the top-left corner displays:
   - Total hands detected
   - Real-time FPS
   - Hand identification (`Left` or `Right`)
   - Count of raised fingers (`0` to `5`)
3. A badge floating near each wrist also displays the quick status.

### How to Exit
- Click on the video window and press the **`Q`** key (or lowercase `q`) to cleanly terminate the program and release webcam resources.

---

## Project Structure

```text
hand-fingers-detection/
│
├── main.py              # Application entrypoint: webcam capture, detection loop, HUD rendering
├── requirements.txt     # Minimal Python dependencies (OpenCV & MediaPipe)
├── README.md            # Project documentation and setup guide
├── .gitignore           # Git ignore file for Python, virtual environments, and IDE artifacts
└── screenshots/         # Directory containing preview images / demo GIFs
    └── .gitkeep
```

---

## Future Improvements

Exciting extensions that can be built on top of this foundation:

- **Gesture recognition**: Recognize common signs such as peace sign, thumbs up, ok, rock-on, or stop.
- **Virtual mouse**: Control cursor movement and clicking actions using finger gestures.
- **Air Canvas**: Draw and paint on the screen in real time using your index fingertip.
- **Sign language recognition**: Classify sign language alphabet (ASL) using machine learning classifiers on landmark embeddings.
- **Gesture-controlled applications**: Control media playback (volume up/down, pause/play, slide transitions) hands-free.

---

## Disclaimer

This is an educational computer-vision project developed for learning, experimentation, and demonstration purposes.
