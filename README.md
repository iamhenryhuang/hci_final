# Hand Gesture Recognition System

Real-time hand gesture recognition using MediaPipe and OpenCV. Detects gestures through your webcam and automatically blurs inappropriate ones. After too many violations, it blurs your face and eventually shuts down the stream.

## Features

- Real-time gesture recognition
- Automatic blurring of inappropriate gestures
- Multi-level penalty system:
  - **Normal**: No penalties
  - **High Warning** (5+ violations): Warning beep + face blur
  - **Shutdown** (10+ violations): Black screen with "STREAM PAUSED"
- Tracks violations daily, auto-resets each day
- Debounce filtering to reduce false detections

## Supported Gestures
**Blocked gestures (will be blurred):**
- `bad!!!` - Thumbs down
- `no!!!` - Middle finger
- `thumb_mid_pinky` - Extended thumb, middle, and pinky (offensive gesture)
- `ok` - OK sign (white supremacist/racist gesture)

## Setup

Requires Python 3.8+:

```bash
python --version
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `mediapipe` - Hand tracking
- `opencv-python` - Image processing
- `numpy` - Numeric operations

## Usage

Go to the `finger_detection` folder and run:

```bash
cd finger_detection
python main.py
```

The camera starts automatically. Inappropriate gestures get blurred immediately. The violation count shows in the top-left corner. Press `q` to quit (resets counter on exit).

## Project Structure

```
hci_final_code/
├── finger_detection/          # Main gesture recognition code
│   ├── main.py                # Entry point
│   ├── gesture_tracker.py     # Tracks gesture counts and daily logs
│   ├── gesture_recognizer.py  # Gesture recognition logic
│   ├── visualizer.py          # Display, blur effects, and stats
│   ├── face_detector.py       # Face detection using MediaPipe
│   ├── geometry.py            # Finger angle calculations
│   └── config.py              # All settings and parameters
├── face_detection/            # Face detection utilities
│   └── face_mosaic.py
└── requirements.txt           # Dependencies
```

## Configuration

Edit `finger_detection/config.py` to customize:

- `CAMERA_INDEX` - Camera device number (default: 0)
- `BAD_GESTURE_THRESHOLD` - Violations before face blur (default: 5)
- `DEBOUNCE_FRAMES` - Frames needed to confirm gesture (default: 3)
- `BLACKLIST_GESTURES` - Which gestures to block
- MediaPipe detection/tracking confidence thresholds
- Mosaic blur levels and display settings

## Notes

- Daily violation counts saved to `gesture_log.json`
- Counters reset automatically each day at midnight
- Pressing `q` resets counter when exiting
- Warning beep works on Windows (winsound), silently ignored on other platforms
- Shutdown mode displays black screen with "STREAM PAUSED" and stops all detection

## Contributors

- 黃柏淵
- 卓俊瑋
- 劉宇盛
- 徐鐽睿
