import cv2
import threading
import time
from PIL import Image
from config import GEMINI_API_KEY, HEADLESS_MODE, tts_queue, NO_DETECT_INTERVAL
from google import genai
from google.genai import types

SYSTEM_PROMPT = (
    "You are a fast embedded assistant for smart glasses for blind users. "
    "Keep replies short, fast and practical."
)


class GeminiSceneDescriber:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set")
        if genai is None or types is None:
            raise RuntimeError("google.genai package is required for scene mode")

        print("[INFO] Loading Gemini scene description mode...")
        self.client = genai.Client()
        self.last_spoken_time = 0
        self.cooldown = 10
        self.last_no_detect_time = 0
        self.completed = False
        self._pending = False
        self._lock = threading.Lock()
        self._generation = 0
        print("[INFO] Gemini scene describer ready.")

    def reset(self):
        self.completed = False
        self.last_spoken_time = 0
        self.last_no_detect_time = 0
        with self._lock:
            self._pending = False
            self._generation += 1

    def process(self, frame):
        if not self.completed:
            now = time.time()
            if now - self.last_spoken_time >= self.cooldown:
                with self._lock:
                    if not self._pending:
                        self._pending = True
                        generation = self._generation
                        worker_frame = frame.copy()
                        threading.Thread(
                            target=self._describe_frame_async,
                            args=(worker_frame, generation),
                            daemon=True,
                        ).start()

        if not HEADLESS_MODE:
            cv2.putText(frame, "Scene Description Mode", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame

    def _describe_frame_async(self, frame, generation):
        description = self._describe_frame(frame)
        with self._lock:
            if generation != self._generation:
                self._pending = False
                return
            self._pending = False

        if description:
            print(f"[SCENE] {description}")
            tts_queue.put(description)
            self.last_spoken_time = time.time()
            self.last_no_detect_time = self.last_spoken_time
            self.completed = True

    def summarize(self, frame):
        return self._describe_frame(frame)

    def _describe_frame(self, frame):
        try:
            img = Image.fromarray(frame, mode="RGB")
            config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    "Provide a concise scene description of this image. "
                    "Describe the main subjects, setting, lighting, colors, and overall mood.",
                    img,
                ],
                config=config
            )
            return response.text.strip()
        except Exception as exc:
            print(f"[ERROR] Gemini scene description failed: {exc}")
            return "Unable to describe the scene right now."
