import os
import queue
import threading
from collections import deque
from pathlib import Path
import time

from dotenv import load_dotenv

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env", override=False)
HEADLESS_MODE = False

# Paths
CURRENCY_MODEL_PATH = str(BASE_DIR / "currency_detection" / "last_15e_ncnn_model")
OBJECT_MODEL_PATH = str(BASE_DIR / "object_detection" / "yolo26n_ncnn_model")
FACE_DB_PATH = str(BASE_DIR / "face_detection" / "encodings_insightface_retina_small.pickle")
OCR_BASE_DIR = BASE_DIR / "ocr"
VOSK_MODEL_PATH = str(BASE_DIR / "vosk-model-small-en-us-0.15")

# Camera
RESOLUTION = (400, 280)


def set_headless_mode(headless: bool):
    """Set headless mode explicitly at runtime. This is intended to be
    called from `main.py` based on CLI args. When headless is True, display is
    disabled. """
    global HEADLESS_MODE
    HEADLESS_MODE = bool(headless)

# CV livestream
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 8000

# Detection
YOLO_CONF = 0.5
FACE_DETECT_INTERVAL = 3
OCR_INTERVAL = 0.8
OCR_MIN_CONFIDENCE = 0.80
FACE_THRESHOLD = 0.35
# Emotion detection config
os.environ["KERAS_BACKEND"] = "torch"
EMOTION_ENABLED = True
EMOTION_MODEL_PATH = str(BASE_DIR / "face_detection" / "emotion.h5")
EMOTION_INPUT_SIZE = (48, 48)
EMOTION_CONFIDENCE_THRESHOLD = 0.20
EMOTION_COOLDOWN = 10.0
EMOTION_DETECT_INTERVAL = FACE_DETECT_INTERVAL  # run emotion inference at same interval by default

# TTS
class _BoundedTTSQueue(queue.Queue):
    """Queue that drops the oldest message when full instead of blocking."""
    def put(self, item, block=True, timeout=None):
        for _ in range(self.maxsize + 1):
            try:
                super().put(item, block=False)
                return
            except queue.Full:
                try:
                    self.get_nowait()
                    self.task_done()  # balance unfinished_tasks for the dropped item
                except queue.Empty:
                    pass

tts_queue = _BoundedTTSQueue(maxsize=20)
TTS_VOICE = "en"
TTS_SPEED = 150
TTS_AMPLITUDE = 160  # espeak-ng amplitude (0-200). Increase for louder output.
MODEL_PATH = "en_US-joe-medium.onnx"
OUTPUT_WAV = "/tmp/tts_output.wav"


# =========================
# MONEY MAPPING
# =========================
MONEY = {
    10: "5 Pounds", 1: "5 Pounds",
    0: "10 Pounds", 3: "10 Pounds",
    9: "20 Pounds", 5: "20 Pounds",
    8: "50 Pounds", 2: "50 Pounds",
    4: "100 Pounds", 11: "100 Pounds",
    6: "200 Pounds", 7: "200 Pounds"
}

# ==========================================
# VOICE CONTROL SHARED STATE
# ==========================================
current_mode = [None]
mode_lock = threading.Lock()

# Lock used to serialize audio device access between TTS and microphone
audio_lock = threading.RLock()

VOICE_COMMANDS = {
    "0": 0, "stop": 0, "exit": 0, "quit": 0,
    "one": 1, "1": 1, "won": 1, "currency": 1,
    "two": 2, "2": 2, "too": 2, "to": 2, "face": 2,
    "three": 3, "3": 3, "free": 3, "tree": 3, "ocr": 3, "text": 3,
    "four": 4, "4": 4, "for": 4, "object": 4,
    "five": 5, "5": 5,
    "six": 6, "6": 6,
    "seven": 7, "7": 7,
    "eight": 8, "8": 8,
    "nine": 9, "9": 9, "color": 9, "colour": 9, "colors": 9,
    "ten": 10, "10": 10, "light": 10, "lights": 10,
}
GEMINI_CHAT_MODE = 5
LOCAL_LLM_CHAT_MODE = 6

MODE_NAMES = {
    1: "Currency Detection",
    2: "Face Recognition",
    3: "OCR/Text Reading",
    4: "Object Detection",
    5: "Gemini Chat Assistant",
    6: "Local LLM Chat Assistant",
    7: "Gemini Scene Description",
    8: "Local LLM Scene Description",
    9: "Color Recognition",
    10: "Light Recognition",
}

# Special (non-mode) voice commands
SPECIAL_COMMANDS = {"help", "repeat", "status", "mode"}

# Shared state for special voice commands
active_mode_ref = [None]   # Updated by main loop so voice can report it
last_spoken_text = [""]    # Updated by TTS worker for "repeat" command

# OCR-specific higher resolution (camera reconfigures on OCR mode switch)
OCR_RESOLUTION = (640, 480)

# Color recognition configuration
COLOR_SAMPLE_SIZE = 40  # pixels. If None, use COLOR_SAMPLE_PERCENT of the frame short side.
COLOR_SAMPLE_PERCENT = 0.05
COLOR_TTS_COOLDOWN = 6.0  # seconds between TTS announcements for color changes
COLOR_SMOOTHING_FRAMES = 5  # temporal smoothing window (frames)

# TTS shutdown sentinel (safer than None — prevents accidental worker death)
TTS_SHUTDOWN = object()

# No-detection heartbeat interval (seconds)
NO_DETECT_INTERVAL = 15

# Thermal monitoring (RPi)
THERMAL_ZONE_PATH = "/sys/class/thermal/thermal_zone0/temp"
THERMAL_WARNING_THRESHOLD = 80  # °C
THERMAL_CHECK_INTERVAL = 30     # seconds

# Face recognition UX tuning
NO_PERSON_GRACE = 2.5       # seconds to wait before announcing "No person detected" after losing sight of a face
PERSON_TTL = 3.5            # seconds to keep a person "active" after last seen (prevents rapid re-announcements when briefly occluded)
ANNOUNCE_EVERY = 15.0       # seconds between announcements of currently seen people (prevents spamming when many people are present or frequently changing)
GREET_COOLDOWN = 30.0       # seconds before re-greeting the same person after they leave and return

# Add these to the bottom of your existing config.py

# =========================================================
# SENSOR & TELEMETRY CONFIG
# =========================================================
FIREBASE_DB_URL = "https://visiosmart2-default-rtdb.firebaseio.com/"
FIREBASE_KEY_PATH = "/home/pi/firebase/serviceAccountKey.json"
DEVICE_ID = "glasses_001"

GPS_SERIAL_PORT = "/dev/ttyACM0"
GPS_BAUD_RATE = 9600

TRIG_PIN = 23
ECHO_PIN = 24
BUZZER_PIN = 18

OBSTACLE_THRESHOLD_M = 0.75
OBSTACLE_THRESHOLD_CM = OBSTACLE_THRESHOLD_M * 100

I2C_BUS = 1
MAX30102_ADDR = 0x57

# Combined Sensor State Tracking Matrix
sensor_state_lock = threading.Lock()
latest_sensor_state = {
    "gps": {
        "latitude": None,
        "longitude": None,
        "speed_knots": None,
        "timestamp_utc": None,
        "status": "no_data",
    },
    "heart": {
        "ok": False,
        "bpm": None,
        "spo2": None,
        "finger": False,
        "ir_dc": None,
        "samples": 0,
        "ts": None,
        "status": "no_signal",
    },
    "obstacle": {
        "distance_cm": None,
        "threshold_cm": OBSTACLE_THRESHOLD_CM,
        "alert": False,
        "ts": None,
        "status": "idle",
    },
    "system": {
        "camera_running": False,
        "camera_available": False,
        "firebase_ok": False,
        "heart_available": False,
        "started_at": int(time.time()),
    }
}



# ==========================================
# LLM (Chat Mode) - Gemini
# ==========================================
LLM_PROVIDER = "gemini"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ==========================================
# LLM (Chat Mode)
# ==========================================
LLM_URL = "http://localhost:8080/v1/chat/completions"
LLM_MODEL = "qwen-chat"
SCENE_MODEL ="smolvlm-vision"
LLM_TIMEOUT_SEC = 90
LLM_MAX_TOKENS = 140
LLM_TEMPERATURE = 0.3
LLM_INENT_TEMPERATURE = 0.0
LLM_TOP_P = 0.8
LLM_TOP_K = 20
LLM_MAX_CONTEXT_CHARS = 1800

FRAMES_TO_CAPTURE = 5  # Number of frames to capture for better detection in chat modes

LLM_CONTEXT_MAX = 6  # short rolling memory for speed
_llm_context_lock = threading.Lock()
_llm_context = deque(maxlen=LLM_CONTEXT_MAX)

# One-shot vision request queue: items are dicts with mode and response_queue
llm_one_shot_queue = queue.Queue(maxsize=2)

def append_llm_context(role, text):
    """Add a short entry to LLM context (role: user/assistant)."""
    if not text:
        return
    with _llm_context_lock:
        print(f"[LLM Context] Appending ({role}): {text}")
        _llm_context.append({"role": role, "content": text.strip()})

def get_llm_context():
    with _llm_context_lock:
        return list(_llm_context)
