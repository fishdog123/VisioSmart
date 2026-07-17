import cv2
import time
from config import COLOR_TTS_COOLDOWN, HEADLESS_MODE, tts_queue


class ColorRecognition:
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

        x1 = max(0, cx - 90)
        x2 = min(w, cx + 90)

        y1 = 45
        y2 = 90

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
                    pass

                self.last_spoken = now

            self.last_label = label

        if not HEADLESS_MODE:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.0
            thickness = 2

            (text_w, text_h), _ = cv2.getTextSize(
                label,
                font,
                scale,
                thickness
            )

            text_x = x1 + (x2 - x1 - text_w) // 2
            text_y = y1 + (y2 - y1 + text_h) // 2

            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                font,
                scale,
                display_bgr,
                thickness
            )

            cv2.circle(frame, (cx, cy), 5, (25, 25, 25), 3)

        return frame

    def summarize(self, frame):
        label, _, _ = self._detect_label_simple(frame)
        return (1, f"Dominant color: {label}")

    def reset(self):
        self.last_label = None
        self.last_spoken = 0.0
