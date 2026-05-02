import cv2
import mediapipe as mp
import time
import os

def draw_landmarks(frame, hand_landmarks):
    h, w, c = frame.shape
    connections = [
        (0,1), (1,2), (2,3), (3,4),
        (0,5), (5,6), (6,7), (7,8),
        (5,9), (9,10), (10,11), (11,12),
        (9,13), (13,14), (14,15), (15,16),
        (13,17), (17,18), (18,19), (19,20),
        (0,17)
    ]
    points = []
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        points.append((cx, cy))
        cv2.circle(frame, (cx, cy), 5, (255, 180, 0), -1)
    for conn in connections:
        pt1, pt2 = points[conn[0]], points[conn[1]]
        cv2.line(frame, pt1, pt2, (0, 220, 180), 2)

def main():
    print("Testing MediaPipe Hands with Tasks API. Show your hand to the camera.")
    print("Press 'q' to quit.")

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    model_path = 'model/hand_landmarker.task'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        num_hands=2,
        running_mode=VisionRunningMode.VIDEO
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Use current time in ms for video mode
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result and result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    draw_landmarks(frame, hand_landmarks)
                cv2.putText(frame, "Hand Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Hand Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('MediaPipe Test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
