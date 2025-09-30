import cv2
import uuid
import time 
import os 
from setup import get_classes

class CaptureImages(): 
    def __init__(self, path: str, classes: list, camera_id: int) -> None: 
        self.cap = cv2.VideoCapture(camera_id) 
        self.path = path 
        self.classes = classes
        
        print("--- Image Capture System Initialized ---")
        
        if not self.cap.isOpened():
            print(f"ERROR: Could not open camera {camera_id}")
            raise Exception(f"Could not open camera {camera_id}")
        else:
            print(f"SUCCESS: Camera {camera_id} connected successfully")
        
        os.makedirs(self.path, exist_ok=True)
        print(f"Output directory set to: {self.path}")

    def capture(self, class_name: str) -> bool:     
        try: 
            ret, frame = self.cap.read() 
            if not ret:
                raise Exception("Failed to read from camera")
                
            raw_frame = frame.copy()
            
            # Display text on the frame for user feedback
            image_text = cv2.putText(frame, f'Capturing for: {class_name}', (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Image Capture', image_text)
            
            # Generate a unique filename and save the image
            filename = f'{class_name}-{uuid.uuid1()}.jpg'
            filepath = os.path.join(self.path, filename)
            cv2.imwrite(filepath, raw_frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("WARNING: 'q' key pressed - stopping capture.")
                return False
                
            return True
            
        except Exception as e: 
            print(f"ERROR capturing for {class_name}: {str(e)}")
            return False

    def run(self, sleep_time: int = 1, num_images: int = 10):
        print("\n--- Starting Capture Session ---")
        print(f"Classes to capture: {self.classes}")
        print(f"Images per class: {num_images}")
        print(f"Delay between captures: {sleep_time} second(s)")
        print("---------------------------------")
        
        total_captured = 0
        
        for img_class in self.classes: 
            print(f"\nStarting capture for class: '{img_class}'")
            class_captured_count = 0
            for i in range(num_images):
                print(f"Capturing image {i+1}/{num_images} for '{img_class}'...")
                success = self.capture(img_class)
                
                if success:
                    class_captured_count += 1
                    total_captured += 1
                else:
                    print(f"Failed to capture image {i+1} for '{img_class}'.")
                
                # Check for quit signal after each attempt
                if cv2.getWindowProperty('Image Capture', cv2.WND_PROP_VISIBLE) < 1:
                    break

                time.sleep(sleep_time)

            print(f"Completed for '{img_class}': {class_captured_count}/{num_images} images captured.")
        
        print("\n--- Capture Session Complete ---")
        print(f"Total images captured across all classes: {total_captured}")
        
        # Clean up resources
        self.cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")

if __name__ == '__main__': 
    CLASSES = get_classes()
    if isinstance(CLASSES, list):
        cap_instance = CaptureImages('./data/images', CLASSES, 0) 
        cap_instance.run(num_images=5, sleep_time=1)
    else:
        print(f"Error loading classes: {CLASSES}")