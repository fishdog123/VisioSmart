import os
from datetime import datetime
from config import RESOLUTION, BASE_DIR
from picamera2 import Picamera2
import cv2


def create_folder(name):
    base_path = BASE_DIR / "face_detection" / "dataset" / name
    base_path.mkdir(parents=True, exist_ok=True)
    return str(base_path)

def capture_photos(name):
    folder = create_folder(name)
    photo_count = 0

    print(f"Taking photos for {name}. Press SPACE to capture, 'q' to quit.")

    while True:
        frame = cam.capture_array()
        cv2.imshow('Capture', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, frame)
            print(f"Photo {photo_count} saved: {filepath}")

        elif key == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()
    print(f"Photo capture completed. {photo_count} photos saved for {name}.")

if __name__ == "__main__":
    cam = Picamera2()
    config = cam.create_video_configuration(main={"format": "BGR888","size": RESOLUTION}, buffer_count=2)
    cam.configure(config)
    cam.start()
    PERSON_NAME = input("Enter the name of the person to capture photos for: ").strip()
    capture_photos(PERSON_NAME)
