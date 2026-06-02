import os
import cv2
import base64
import requests
import threading
import time
from config import (
    tts_queue,
    HEADLESS_MODE,
    SCENE_MODEL,
    LLM_URL,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_TOP_K,
    LLM_TIMEOUT_SEC
)

SCENE_VLM_PROMPT = (
    "You are the real-time audio description assistant for a blind user wearing smart glasses.\n\n"
    "CRITICAL DIRECTIVES:\n"
    "1. Objective & Factual: Describe only what is clearly visible. Do not guess, assume, or interpret. Avoid bias or emotional commentary.\n"
    "2. Spatial Navigation: Prioritize layout, immediate path obstacles, nearby people, and major objects directly ahead. State their spatial locations relative to the user.\n"
    "3. Trait Consistency: When describing people, prioritize factual visible traits (e.g., 'a bearded man'). Never assume or guess racial, ethnic, or gender identities if not explicitly evident.\n"
    "4. Read Text: If text or signs are central to understanding the immediate scene, read the text verbatim, prefixed by 'Sign reads:'.\n"
    "5. Style: Speak in the present tense, third-person, using concise, vivid, and formal language. Be highly succinct.\n\n"
    "TASK:\n"
    "Provide a clear, 2-to-3 sentence spatial summary of the scene ahead. No conversational filler."
)

class LocalSceneDescriber:
    def __init__(self):
        """Initializes the VLM Scene Description module."""
        print("[SCENE] SmolVLM Scene Describer initialized.")
        self.last_spoken_time = 0
        self.cooldown = 10
        self.last_no_detect_time = 0
        self.completed = False
        self._pending = False
        self._lock = threading.Lock()

    def reset(self):
        """Resets the state of the describer for clean mode-switching."""
        self.completed = False
        self.last_spoken_time = 0
        self.last_no_detect_time = 0
        with self._lock:
            self._pending = False

    def process(self, frame):
        """
        Evaluates the scene non-blockingly. If cooldown has passed and no inference
        is active, it dispatches a background worker thread so the camera stream doesn't freeze.
        """
        if not self.completed:
            now = time.time()
            if now - self.last_spoken_time >= self.cooldown:
                with self._lock:
                    if not self._pending:
                        self._pending = True
                        # Shallow copy to prevent mutations while thread runs
                        worker_frame = frame.copy()
                        threading.Thread(
                            target=self._describe_frame_async,
                            args=(worker_frame,),
                            daemon=True,
                        ).start()

        if not HEADLESS_MODE:
            cv2.putText(frame, "Local Scene Description Mode", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame

    def _describe_frame_async(self, frame):
        """Background thread worker target."""
        description = self._describe_frame(frame)
        with self._lock:
            self._pending = False

        if description:
            print(f"[SCENE] {description}")
            tts_queue.put(description)
            self.last_spoken_time = time.time()
            self.last_no_detect_time = self.last_spoken_time
            self.completed = True

    def summarize(self, frame):
        """Synchronous helper function matching Gemini describer design."""
        return self._describe_frame(frame)

    def _describe_frame(self, frame):
        """Handles file serialization, base64 compilation, and local VLM routing."""
        image_path = "/tmp/vlm_scene_capture.jpg"

        try:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, frame_bgr)
        except Exception as e:
            print(f"[SCENE] Failed to write matrix frame to file partition: {e}")
            return "Camera image storage failure."

        if not os.path.exists(image_path):
            return "Failed to find captured image."

        try:
            print("[VLM] Packaging image payload for SmolVLM...")
            image_data_url = self._get_image_base64_url(image_path)

            messages = [
                {
                    "role": "system",
                    "content": SCENE_VLM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this scene for me."},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url}
                        }
                    ]
                }
            ]

            payload = {
                "model": SCENE_MODEL,
                "messages": messages,
                "temperature": LLM_TEMPERATURE,
                "top_p": LLM_TOP_P,
                "top_k": LLM_TOP_K,
            }

            print(f"[VLM] Sending POST request to local endpoint: {LLM_URL}")
            response = requests.post(LLM_URL, json=payload, timeout=LLM_TIMEOUT_SEC)
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            print(f"[VLM] Raw Description: {content}")

            # Unpack accidental structural JSON returns if your small VLM outputs quirks
            if content.startswith("{"):
                start = content.find('"text":')
                if start != -1:
                    sub = content[start+7:]
                    end = sub.find('"')
                    if end != -1:
                        return sub[:end]

            return content

        except Exception as exc:
            print(f"[ERROR] Local scene description failed: {exc}")
            return "Local scene description server connection error."

    def _get_image_base64_url(self, image_path):
        """Converts a local image file into a base64 data URL for the VLM payload."""
        with open(image_path, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_data}"
