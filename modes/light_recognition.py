import cv2
import time
import numpy as np

from config import COLOR_TTS_COOLDOWN, HEADLESS_MODE, tts_queue


class LightRecognition:
    """Simple ambient light detector using BGR->GRAY average brightness.

    Public API:
    - process(frame) -> annotated frame
    - summarize(frame) -> short text summary
    - reset()
    """

    def __init__(self):
        self.last_label = None
        self.last_spoken = 0.0
        self.last_value = 0.0
        self.cooldown = COLOR_TTS_COOLDOWN or 6.0

    def _avg_brightness(self, frame):
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        # sample a small center square to reduce noise
        size = max(8, int(min(h, w) * 0.05))
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - size // 2)
        x2 = min(w, cx + size // 2)
        y1 = max(0, cy - size // 2)
        y2 = min(h, cy + size // 2)
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())

    def _label_for_brightness(self, avg):
        if avg is None:
            return "Unknown"
        if avg < 60:
            return "DARK"
        if avg < 180:
            return "MEDIUM"
        return "BRIGHT"

    def process(self, frame):
        avg = self._avg_brightness(frame)
        label = self._label_for_brightness(avg)
        now = time.time()
        if label != self.last_label:
            if now - self.last_spoken > (self.cooldown or 6.0):
                try:
                    tts_queue.put(f"Light: {label}")
                except Exception:
                    pass
                self.last_spoken = now
            self.last_label = label
        self.last_value = avg or 0.0

        if not HEADLESS_MODE and frame is not None and frame.size != 0:
            h, w = frame.shape[:2]
            # draw overlay
            cv2.rectangle(frame, (10, 10), (300, 90), (255, 255, 255), -1)
            txt = f"Light: {label} ({int(self.last_value)})"
            cv2.putText(frame, txt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        return frame

    def summarize(self, frame):
        avg = self._avg_brightness(frame)
        label = self._label_for_brightness(avg)
        if avg is None:
            return (0, "Lighting: unknown")
        return (1, f"Lighting: {label} ({int(avg)})")

    def reset(self):
        self.last_label = None
        self.last_spoken = 0.0
        self.last_value = 0.0
