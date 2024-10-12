import cv2
import mediapipe as mp
import numpy as np
import random
import time

class WhackAMole:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.score = 0
        self.mole_pos = None
        self.mole_appear_time = 0
        self.mole_duration = 2  # seconds
        self.mole_size = 60  # diameter of the mole

        self.start_time = time.time()
        self.elapsed_time = 0
        self.paused = False
        self.pause_start_time = 0

    def create_mole(self, frame, x, y):
        cv2.circle(frame, (x, y), self.mole_size // 2, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), self.mole_size // 2, (0, 0, 0), 2)
        eye_size = self.mole_size // 10
        cv2.circle(frame, (x - self.mole_size // 6, y - self.mole_size // 6), eye_size, (0, 0, 0), -1)
        cv2.circle(frame, (x + self.mole_size // 6, y - self.mole_size // 6), eye_size, (0, 0, 0), -1)
        cv2.ellipse(frame, (x, y + self.mole_size // 6), (self.mole_size // 5, self.mole_size // 10), 0, 0, 180, (0, 0, 0), 2)

    def generate_mole_position(self):
        x = random.randint(self.mole_size, self.width - self.mole_size)
        y = random.randint(self.mole_size, self.height - self.mole_size)
        return (x, y)

    def draw_score(self, frame):
        cv2.putText(frame, f"Score: {self.score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def draw_chronometer(self, frame):
        if not self.paused:
            self.elapsed_time = int(time.time() - self.start_time)
        cv2.putText(frame, f"Time: {self.elapsed_time}s", (self.width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def check_hit(self, hand_landmarks):
        if self.mole_pos is None:
            return False
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x, y = int(index_finger_tip.x * self.width), int(index_finger_tip.y * self.height)
        mole_x, mole_y = self.mole_pos
        distance = np.sqrt((x - mole_x)**2 + (y - mole_y)**2)
        return distance < self.mole_size // 2

    def reset_game(self):
        self.score = 0
        self.mole_pos = None
        self.start_time = time.time()
        self.elapsed_time = 0
        self.paused = False

    def toggle_pause(self):
        if self.paused:
            self.start_time += time.time() - self.pause_start_time
        else:
            self.pause_start_time = time.time()
        self.paused = not self.paused

    def run(self):
        last_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            current_time = time.time()
            if not self.paused:
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        if self.check_hit(hand_landmarks):
                            self.score += 1
                            self.mole_pos = None

                if self.mole_pos is None or current_time - self.mole_appear_time > self.mole_duration:
                    self.mole_pos = self.generate_mole_position()
                    self.mole_appear_time = current_time

                if self.mole_pos:
                    self.create_mole(frame, *self.mole_pos)

            self.draw_score(frame)
            self.draw_chronometer(frame)

            if self.paused:
                cv2.putText(frame, "PAUSED", (self.width // 2 - 70, self.height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(frame, "q: quit, r: restart, p: pause/resume", (10, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Whack-a-Mole', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_game()
            elif key == ord('p'):
                self.toggle_pause()

            if not self.paused:
                fps = 1 / (current_time - last_time)
                last_time = current_time
                cv2.putText(frame, f"FPS: {fps:.2f}", (self.width - 120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = WhackAMole()
    game.run()