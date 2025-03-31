import  cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import screen_brightness_control as sbc


class HandControlSystem:
    def __init__(self):
        # Constants and Configuration
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIN_DETECTION_CONFIDENCE = 0.7
        self.MIN_TRACKING_CONFIDENCE = 0.7

        # State variables
        self.prev_distance = 0
        self.smoothing_factor = 0.5
        self.last_action_time = 0
        self.cooldown_period = 0.1
        self.mode = "volume"

        # UI Colors and Font
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_color = (255, 255, 255)
        self.volume_color = (0, 165, 255)  # Orange
        self.brightness_color = (255, 255, 0)  # Cyan

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize Webcam
        self.cam = cv2.VideoCapture(0)
        if not self.cam.isOpened():
            raise Exception("Could not open webcam. Please check your camera.")

        print("Hand Gesture Control System Started")
        print("===================================")
        print("Press 'm' to switch between Volume and Brightness control")
        print("Press 'q' to quit the application")

    def calculate_distance(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    def smooth_distance(self, current_distance):
        smoothed = self.smoothing_factor * current_distance + (1 - self.smoothing_factor) * self.prev_distance
        self.prev_distance = smoothed
        return smoothed

    def adjust_volume(self, distance):
        current_time = time.time()
        if current_time - self.last_action_time < self.cooldown_period:
            return

        # Map distance to volume range
        volume_level = np.interp(distance, [30, 300], [0, 100])
        if volume_level > 50:
            pyautogui.press("volumeup")
        else:
            pyautogui.press("volumedown")

        self.last_action_time = current_time

    def adjust_brightness(self, distance):
        current_time = time.time()
        if current_time - self.last_action_time < self.cooldown_period:
            return

        # Map distance to brightness range
        brightness_level = int(np.interp(distance, [30, 300], [0, 100]))

        try:
            sbc.set_brightness(brightness_level)
            self.last_action_time = current_time
        except Exception as e:
            print(f"Failed to adjust brightness: {e}")

    def toggle_mode(self):
        self.mode = "brightness" if self.mode == "volume" else "volume"
        print(f"Switched to {self.mode.upper()} mode")

    def draw_ui(self, image, landmarks, finger_positions):
        """Draw UI elements on the frame"""
        thumb_pos = finger_positions[self.THUMB_TIP]
        index_pos = finger_positions[self.INDEX_TIP]

        # Draw circle at the tips
        cv2.circle(image, thumb_pos, 12, (0, 0, 255), -1)  # Red for thumb
        cv2.circle(image, index_pos, 12, (0, 255, 255), -1)  # Yellow for index

        # Draw line between thumb and index finger
        color = self.volume_color if self.mode == "volume" else self.brightness_color
        cv2.line(image, thumb_pos, index_pos, color, 3)

        # Calculate distance and display it
        distance = self.calculate_distance(thumb_pos, index_pos)
        cv2.putText(image, f"Distance: {int(distance)}", (10, 30),
                    self.font, 0.7, self.text_color, 2)

        # Display mode
        mode_text = f"MODE: {self.mode.upper()}"
        cv2.putText(image, mode_text, (10, 70),
                    self.font, 0.9, color, 2)

        # Draw control bar
        bar_value = np.interp(distance, [30, 300], [400, 150])
        cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
        cv2.rectangle(image, (50, int(bar_value)), (85, 400), color, -1)
        cv2.putText(image, f"{int(np.interp(distance, [30, 300], [0, 100]))}%",
                    (50, 430), self.font, 0.7, self.text_color, 2)

        return distance

    def process_landmarks(self, hand_landmarks, shape):
        """Convert hand landmarks into pixel coordinates"""
        positions = {}
        h, w, _ = shape
        for idx, landmark in enumerate(hand_landmarks.landmark):
            x, y = int(landmark.x * w), int(landmark.y * h)
            positions[idx] = (x, y)
        return positions

    def run(self):
        """Main execution loop"""
        try:
            while True:
                ret, frame = self.cam.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                # Flip the frame horizontally
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process hands
                results = self.hands.process(rgb_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.toggle_mode()

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        positions = self.process_landmarks(hand_landmarks, frame.shape)
                        distance = self.draw_ui(frame, hand_landmarks, positions)
                        smoothed_distance = self.smooth_distance(distance)

                        if self.mode == "volume":
                            self.adjust_volume(smoothed_distance)
                        else:
                            self.adjust_brightness(smoothed_distance)

                cv2.imshow("Hand Control System", frame)

        finally:
            self.cam.release()
            cv2.destroyAllWindows()
            self.hands.close()
            print("Application closed.")


if __name__ == "__main__":
    try:
        controller = HandControlSystem()
        controller.run()
    except Exception as e:
        print(f"Error: {e}")
