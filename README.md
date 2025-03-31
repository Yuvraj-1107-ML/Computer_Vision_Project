# Hand Gesture Control System

## Overview
The **Hand Gesture Control System** is a computer vision-based project that enables users to interact with a computer using hand gestures. It utilizes OpenCV and MediaPipe for real-time hand tracking and gesture recognition, allowing hands-free control of various applications.

## Features
- Real-time hand tracking using OpenCV and MediaPipe
- Gesture recognition for various actions
- Control system functionalities using predefined hand gestures
- User-friendly and interactive interface

## Technologies Used
- **Python**
- **OpenCV** (Computer Vision Library)
- **MediaPipe** (Hand Tracking and Pose Estimation)
- **NumPy** (Scientific Computing)
- **PyAutoGUI** (Simulating Keyboard and Mouse Inputs)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/hand-gesture-control.git
   cd hand-gesture-control
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the program:
   ```bash
   python hand_gesture_control.py
   ```

## How It Works
1. The webcam captures real-time video input.
2. MediaPipe detects and tracks hand landmarks.
3. Custom gestures are mapped to specific system commands.
4. Using PyAutoGUI, gestures trigger corresponding actions (e.g., volume control, mouse movement, media playback).

## Usage
- **Two-Finger Pinch**: Zoom In/Out

## Future Improvements
- Add support for more complex gestures.
- Integrate with smart home devices.
- Improve accuracy using machine learning models.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.



