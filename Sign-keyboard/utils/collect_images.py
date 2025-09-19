import cv2
import numpy as np
import uuid
import time
import os
from setup import get_classes

# Assuming get_classes() returns a list of class names
classes = get_classes()

class CaptureImages():
    def __init__(self, path: str, classes: list, camera_id: int) -> None:
        self.cap = cv2.VideoCapture(camera_id)
        self.path = path
        self.classes = classes

        # Verify camera connection
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            raise Exception(f"Could not open camera {camera_id}")
        else:
            print(f"Camera {camera_id} connected successfully")

        # Ensure output directory exists
        os.makedirs(self.path, exist_ok=True)
        print(f"Output directory set to: {self.path}")

    def capture(self, class_name: str) -> bool:
        try:
            ret, frame = self.cap.read()
            raw_frame = frame.copy()
            if not ret:
                raise Exception("Failed to read from camera")

            # Display the class name on the frame
            image = cv2.putText(frame, f'Capturing {class_name}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Image Capture', image)

            # Generate unique filename and save the original frame
            filename = f'{class_name}-{uuid.uuid1()}.jpg'
            filepath = os.path.join(self.path, filename)
            cv2.imwrite(filepath, raw_frame)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit key pressed - stopping capture.")
                return False

            return True

        except Exception as e:
            print(f"Error capturing image for class '{class_name}': {e}")
            return False

    def run(self, sleep_time: int = 1, num_images: int = 10):
        print("\n--- Starting Image Capture Session ---")
        print(f"Classes: {', '.join(self.classes)}")
        print(f"Images per class: {num_images}")
        print(f"Delay between captures: {sleep_time} second(s)")
        print("--------------------------------------\n")

        total_captured = 0

        for class_name in self.classes:
            print(f"Starting capture for class: '{class_name}'")
            # Countdown before starting
            for i in range(5, 0, -1):
                print(f"Starting in {i}...")
                time.sleep(1)

            class_captured = 0
            for idx in range(num_images):
                print(f"Capturing image {idx + 1}/{num_images} for '{class_name}'...")
                success = self.capture(class_name)

                if success:
                    class_captured += 1
                    total_captured += 1
                else:
                    # If capture returns False due to 'q' press, exit the loop
                    break

                time.sleep(sleep_time)
            
            print(f"Completed '{class_name}': {class_captured}/{num_images} images captured.\n")

            # Check if we need to exit the main loop as well
            if class_captured < num_images and not self.capture(class_name):
                 break


        print("--- Capture Session Complete ---")
        print(f"Total images captured: {total_captured}")
        print("--------------------------------\n")

        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")

if __name__ == '__main__':
    cap = CaptureImages('./data/test', classes, 0)

    # Start the capture run
    cap.run(num_images=5, sleep_time=1)