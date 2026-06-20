import cv2


def main():
    print("Testing Webcam. Press 'q' to quit.")

    # 0 is usually the default built-in webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Display the frame
        cv2.imshow('Webcam Test', frame)

        # Wait for 'q' key to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release and close
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
