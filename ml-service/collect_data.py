import cv2
import time
import mediapipe as mp
import csv
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
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Make sure model exists
    model_path = 'model/hand_landmarker.task'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Ensure it was downloaded.")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        num_hands=1,
        running_mode=VisionRunningMode.VIDEO
    )

    os.makedirs('data', exist_ok=True)
    csv_file_path = 'data/chord_data.csv'
    write_header = not os.path.exists(csv_file_path)
    
    csv_file = open(csv_file_path, 'a', newline='')
    writer = csv.writer(csv_file)
    if write_header:
        header = ['label'] + [f'{axis}_{i}' for i in range(21) for axis in ['x', 'y', 'z']]
        writer.writerow(header)

    key_to_chord = {
        ord('a'): 'Am', ord('c'): 'C', ord('d'): 'D',
        ord('e'): 'E', ord('m'): 'Em', ord('f'): 'F',
        ord('g'): 'G', ord('b'): 'Bm', ord('s'): 'F#m',
        ord('7'): 'Fmaj7',
    }

    cap = cv2.VideoCapture(0)
    print("\nGuitar Chord Data Collector (Tasks API)")
    print("Press a chord key to save landmarks. Press 'q' to quit.")
    print("Keys:", {chr(k): v for k, v in key_to_chord.items()})

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

            landmarks_flat = []
            if result and result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    draw_landmarks(frame, hand_landmarks)
                    for lm in hand_landmarks:
                        landmarks_flat.extend([lm.x, lm.y, lm.z])

            cv2.putText(frame, "Press key to save. 'q' to quit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Data Collector', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in key_to_chord:
                if landmarks_flat:
                    label = key_to_chord[key]
                    writer.writerow([label] + landmarks_flat)
                    csv_file.flush()
                    print(f"Saved sample for chord: {label}")
                else:
                    print("Hand not detected! Cannot save.")

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

if __name__ == '__main__':
    main()
