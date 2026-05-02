import cv2
import mediapipe as mp
import csv
import os

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # Setup directories and CSV file
    os.makedirs('data', exist_ok=True)
    csv_file_path = 'data/chord_data.csv'
    
    # Check if we need to write the header
    write_header = not os.path.exists(csv_file_path)
    
    csv_file = open(csv_file_path, 'a', newline='')
    writer = csv.writer(csv_file)
    
    if write_header:
        header = ['label']
        for i in range(21):
            header.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
        writer.writerow(header)

    # Chord labels mapped to keys
    key_to_chord = {
        ord('a'): 'Am',
        ord('c'): 'C',
        ord('d'): 'D',
        ord('e'): 'E',
        ord('m'): 'Em',
        ord('f'): 'F',
        ord('g'): 'G',
        ord('b'): 'Bm',
        ord('s'): 'F#m',
        ord('7'): 'Fmaj7',
    }

    # Open Webcam
    cap = cv2.VideoCapture(0)
    print("Webcam started. Press a chord key to save landmarks. Press 'q' to quit.")
    print("Mapped keys:", {chr(k): v for k, v in key_to_chord.items()})

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and find hands
        results = hands.process(rgb_frame)

        landmarks_flat = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Flatten the 21 landmarks into a 63-element list
                for lm in hand_landmarks.landmark:
                    landmarks_flat.extend([lm.x, lm.y, lm.z])

        # Display the resulting frame
        cv2.putText(frame, "Press key to save. 'q' to quit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Guitar Chord Data Collector', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in key_to_chord and results.multi_hand_landmarks:
            label = key_to_chord[key]
            # Save to CSV
            row = [label] + landmarks_flat
            writer.writerow(row)
            csv_file.flush()
            print(f"Saved sample for chord: {label}")
        elif key in key_to_chord and not results.multi_hand_landmarks:
            print("Hand not detected! Cannot save.")

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

if __name__ == '__main__':
    main()
