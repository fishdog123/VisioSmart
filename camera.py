import threading

from config import RESOLUTION, tts_queue

# ==========================================
# CAMERA
# ==========================================
cam = None
_current_resolution = None
_cam_lock = threading.Lock()

try:
    from picamera2 import Picamera2
    cam = Picamera2()
    config = cam.create_video_configuration(main={"format": "BGR888","size": RESOLUTION}, buffer_count=2)
    cam.configure(config)
    cam.start()
    _current_resolution = RESOLUTION
    print("[INFO] Camera started successfully.")
except Exception as e:
    print(f"[ERROR] Camera initialization failed: {e}")
    tts_queue.put("Error: Camera failed to start. Please check the connection.")

def get_frame():
    if cam is None:
        return None
    with _cam_lock:
        try:
            return cam.capture_array()
        except Exception as e:
            print(f"[ERROR] Frame capture failed: {e}")
            return None

def reconfigure_camera(resolution):
    """Reconfigure camera resolution (e.g., higher res for OCR mode)."""
    global cam, _current_resolution
    if cam is None or resolution == _current_resolution:
        return
    with _cam_lock:
        try:
            cam.stop()
            config = cam.create_video_configuration(
                main={"format": "BGR888", "size": resolution}, buffer_count=2
            )
            cam.configure(config)
            cam.start()
            _current_resolution = resolution
            print(f"[INFO] Camera reconfigured to {resolution[0]}x{resolution[1]}")
        except Exception as e:
            print(f"[WARNING] Camera reconfigure to {resolution} failed: {e}")
            try:
                config = cam.create_video_configuration(
                    main={"format": "BGR888", "size": _current_resolution}, buffer_count=2
                )
                cam.configure(config)
                cam.start()
            except Exception:
                print("[ERROR] Camera restart failed after reconfigure error")

def release_camera():
    if cam is not None:
        try:
            cam.stop()
        except Exception:
            pass
