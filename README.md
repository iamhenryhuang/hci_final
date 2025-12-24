# Hand Gesture Recognition System

A real-time hand gesture recognition system built with MediaPipe and OpenCV. It detects gestures through your webcam and automatically blurs inappropriate gestures. After too many violations, it'll also blur your face.

## What it does

- Recognizes hand gestures in real-time (thumbs up, thumbs down, OK sign, rock, fist, etc.)
- Automatically blurs inappropriate gestures
- Keeps track of violations - after 5 bad gestures, it blurs your face too
- After 10 violations, the stream pauses and shows a black screen

## Supported gestures

**Normal gestures:**
- `good` - Thumbs up
- `ROCK!` - Rock hand sign
- `fist` - Fist

**Blocked gestures (will be blurred):**
- `bad!!!` - Thumbs down
- `no!!!` - Middle finger
- `thumb_mid_pinky` - Thumb + middle + pinky
- `ok` - OK sign

## Setup

Make sure you have Python 3.8 or higher:

```bash
python --version
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:
- `mediapipe` - Hand tracking
- `opencv-python` - Image processing
- `numpy` - Number crunching

## Usage

Navigate to the `finger_detection` folder and run:

```bash
cd finger_detection
python main.py
```

The camera will start automatically. Inappropriate gestures get blurred, and the violation count shows in the top-left corner. Press `q` to quit.

## Project structure

```
hci_final_code/
├── finger_detection/          # Main gesture recognition code
│   ├── main.py                # Entry point
│   ├── gesture_tracker.py     # Tracks gesture counts
│   ├── gesture_recognizer.py  # Gesture recognition logic
│   ├── visualizer.py          # Display and blur effects
│   ├── face_detector.py       # Face detection
│   ├── geometry.py            # Finger angle calculations
│   └── config.py              # Settings and parameters
├── face_detection/            # Face detection module
│   └── face_mosaic.py
└── requirements.txt           # Dependencies
```

## Configuration

All parameters can be tweaked in `finger_detection/config.py`:

- `CAMERA_INDEX` - Camera device number (default: 0)
- `BAD_GESTURE_THRESHOLD` - Number of violations before face blur kicks in (default: 5)
- `DEBOUNCE_FRAMES` - Frames required to confirm a gesture (default: 3)
- Other MediaPipe and display settings

## Notes

- Daily violation counts are logged to `gesture_log.json`
- Counters reset automatically each day
- Pressing `q` resets the counter when you exit
