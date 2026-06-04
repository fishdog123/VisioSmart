import cv2
import time

from config import COLOR_TTS_COOLDOWN, HEADLESS_MODE, tts_queue


class ColorRecognition:
    """Minimal color recognition: center-pixel HSV mapping.

    Keeps the same public API used by other modes:
    - `process(frame)` returns an annotated frame
    - `summarize(frame)` returns a short text summary
    - `reset()` clears internal state
    """

    def __init__(self):
        self.last_label = None
        self.last_spoken = 0.0

    def _detect_label_simple(self, frame):
        if frame is None or frame.size == 0:
            return "Unknown", (255, 255, 255), (0, 0, 0, 0, 0, 0)

        h, w = frame.shape[:2]
        cx = w // 2
        cy = h // 2

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        pixel = hsv[cy, cx]
        hue = int(pixel[0])
        sat = int(pixel[1])
        val = int(pixel[2])

        # Detect black (low value) and white (low saturation + high value)
        if val < 50:
            label = "BLACK"
        elif sat < 30 and val > 220:
            label = "WHITE"
        else:
            label = "Undefined"
            if hue < 5:
                label = "RED"
            elif hue < 22:
                label = "ORANGE"
            elif hue < 33:
                label = "YELLOW"
            elif hue < 78:
                label = "GREEN"
            elif hue < 131:
                label = "BLUE"
            elif hue < 170:
                label = "VIOLET"
            else:
                label = "RED"

        bgr = frame[cy, cx]
        display_bgr = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

        x1 = max(0, cx - 220)
        x2 = min(w, cx + 200)
        y1 = 10
        y2 = 120

        return label, display_bgr, (x1, y1, x2, y2, cx, cy)

    def process(self, frame):
        label, display_bgr, box = self._detect_label_simple(frame)
        x1, y1, x2, y2, cx, cy = box

        now = time.time()
        if label != self.last_label:
            if now - self.last_spoken > (COLOR_TTS_COOLDOWN or 6.0):
                try:
                    tts_queue.put(f"Color: {label}")
                except Exception:
                    # If TTS queue isn't available, don't crash the pipeline
                    pass
                self.last_spoken = now
            self.last_label = label

        if not HEADLESS_MODE:
            # Draw a filled white rectangle and the large label (matching color.py)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
            cv2.putText(frame, label, (cx - 200, 100), 0, 3, display_bgr, 5)
            cv2.circle(frame, (cx, cy), 5, (25, 25, 25), 3)

        return frame

    def summarize(self, frame):
        label, _, _ = self._detect_label_simple(frame)
        return (1, f"Dominant color: {label}")

    def reset(self):
        self.last_label = None
        self.last_spoken = 0.0
