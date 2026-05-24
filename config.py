import atexit
import os
import queue
import sys
import threading
from collections import deque
from pathlib import Path

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
STREAM_HOST="0.0.0.0"
STREAM_PORT="8081"

# Detection
YOLO_CONF = 0.5
FACE_DETECT_INTERVAL = 3
OCR_INTERVAL = 0.8
OCR_MIN_CONFIDENCE = 0.80
FACE_THRESHOLD = 0.35

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
# VOICE_COMMANDS = {
#     "one": 1, "won": 1,
#     "two": 2, "to": 2, "too": 2,
#     "three": 3, "free": 3, "tree": 3,
#     "four": 4, "for": 4,
#     "stop": 0, "exit": 0, "quit": 0,
# }

CHAT_MODE = 5

VOICE_COMMANDS = {
    "one": 1, "won": 1, "currency": 1,
    "two": 2, "too": 2, "to": 2, "face": 2,
    "three": 3, "free": 3, "tree": 3, "ocr": 3, "text": 3,
    "four": 4, "for": 4, "object": 4,
    "five": 5, "chat": 5, "assistant": 5,
    "six": 6, "scene": 6, "describe": 6,
    "stop": 0, "exit": 0, "quit": 0,
}
MODE_NAMES = {
    1: "Currency Detection",
    2: "Face Recognition",
    3: "OCR/Text Reading",
    4: "Object Detection",
    5: "Chat Assistant",
    6: "Scene Description",
}

# Special (non-mode) voice commands
# SPECIAL_COMMANDS = {"help", "repeat", "status", "mode"}
SPECIAL_COMMANDS ={}

# Shared state for special voice commands
active_mode_ref = [None]   # Updated by main loop so voice can report it
last_spoken_text = [""]    # Updated by TTS worker for "repeat" command

# OCR-specific higher resolution (camera reconfigures on OCR mode switch)
OCR_RESOLUTION = (640, 480)

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

# ==========================================
# LLM (Chat Mode) - Gemini
# ==========================================
LLM_PROVIDER = "gemini"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


LLM_TIMEOUT_SEC = 30
LLM_MAX_TOKENS = 140
LLM_TEMPERATURE = 0.6
LLM_TOP_P = 0.8
LLM_TOP_K = 20
LLM_MAX_CONTEXT_CHARS = 1800

LLM_CONTEXT_MAX = 14  # short rolling memory for speed
_llm_context_lock = threading.Lock()
_llm_context = deque(maxlen=LLM_CONTEXT_MAX)

# One-shot vision request queue: items are dicts with mode and response_queue
llm_one_shot_queue = queue.Queue(maxsize=2)

def append_llm_context(role, text):
    """Add a short entry to LLM context (role: user/assistant)."""
    if not text:
        return
    with _llm_context_lock:
        _llm_context.append({"role": role, "content": text.strip()})

def get_llm_context():
    with _llm_context_lock:
        return list(_llm_context)
