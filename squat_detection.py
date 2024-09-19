import cv2
import mediapipe as mp
import numpy as np
import time
import os

# Initialize Mediapipe Pose class
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to determine if the squat is correct based on angles and positioning
def is_squat_correct(knee_angle, hip_knee_angle, ankle_knee_position_diff):
    if knee_angle < 90 and hip_knee_angle < 100 and abs(ankle_knee_position_diff) < 0.1:
        return True
    return False

# Function to detect squats and provide feedback with visual indicators
def detect_squats_optimized(video_source, target_resolution=(1280, 720)):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_source}.")
        return

    squat_count = 0
    stage = None
    prev_stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, target_resolution)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                hip_knee_angle = calculate_angle(hip, knee, ankle)
                knee_angle = calculate_angle([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y], hip, knee)
                ankle_knee_position_diff = ankle[0] - knee[0]

                correct_squat = is_squat_correct(knee_angle, hip_knee_angle, ankle_knee_position_diff)
                color = (0, 255, 0) if correct_squat else (0, 0, 255)

                if hip_knee_angle > 160:
                    stage = "up"
                if hip_knee_angle < 90 and stage == 'up':
                    stage = "down"
                    if prev_stage != "down":
                        squat_count += 1
                        prev_stage = "down"
                    print(f"Squat Count: {squat_count}")

                cv2.putText(image, f'Squats: {squat_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.putText(image, "Incorrect" if not correct_squat else "Correct", (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Error: {e}")

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Squat Detection Optimized', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
