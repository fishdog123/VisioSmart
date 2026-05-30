import cv2
import numpy as np
import time
from collections import deque, Counter

from config import (
    COLOR_SAMPLE_SIZE, COLOR_SAMPLE_PERCENT, COLOR_SMOOTHING_FRAMES,
    COLOR_TTS_COOLDOWN, COLOR_BLACK_L_THRESHOLD, COLOR_WHITE_L_THRESHOLD,
    COLOR_GRAY_CHROMA, HEADLESS_MODE, tts_queue
)


class ColorRecognition:
    def __init__(self):
        # Palette defined in RGB for readability; we'll convert to BGR/Lab
        self.palette_rgb = {
            "Red": (255, 0, 0),
            "Orange": (255, 165, 0),
            "Yellow": (255, 255, 0),
            "Green": (0, 128, 0),
            "Blue": (0, 0, 255),
            "Purple": (128, 0, 128),
            "Pink": (255, 192, 203),
            "Brown": (150, 75, 0),
            "Black": (0, 0, 0),
            "White": (255, 255, 255),
            "Gray": (128, 128, 128),
        }

        # Precompute Lab values for perceptual distance and store display BGR
        self.palette_lab = {}
        self.palette_bgr = {}
        for name, rgb in self.palette_rgb.items():
            # OpenCV expects BGR ordering
            bgr = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0, :].astype(float)
            self.palette_lab[name] = lab
            self.palette_bgr[name] = (int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2]))

        self.buffer = deque(maxlen=(COLOR_SMOOTHING_FRAMES or 5))
        self.last_label = None
        self.last_spoken = 0.0

    def _compute_sample_box(self, frame):
        h, w = frame.shape[:2]
        if COLOR_SAMPLE_SIZE and COLOR_SAMPLE_SIZE > 0:
            size = int(COLOR_SAMPLE_SIZE)
        else:
            size = max(8, int(min(w, h) * (COLOR_SAMPLE_PERCENT or 0.05)))
        half = size // 2
        cx = w // 2
        cy = h // 2
        x1 = max(0, cx - half)
        x2 = min(w, cx + half)
        y1 = max(0, cy - half)
        y2 = min(h, cy + half)
        return x1, y1, x2, y2, cx, cy

    def _detect_label(self, frame):
        x1, y1, x2, y2, cx, cy = self._compute_sample_box(frame)
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return "Unknown", (255, 255, 255), (x1, y1, x2, y2, cx, cy)

        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        mean_lab = lab.reshape(-1, 3).mean(axis=0)
        L, a, b = mean_lab
        chroma = ((a - 128) ** 2 + (b - 128) ** 2) ** 0.5

        # Heuristics for black/white/gray
        if L <= (COLOR_BLACK_L_THRESHOLD or 20):
            return "Black", self.palette_bgr["Black"], (x1, y1, x2, y2, cx, cy)
        if L >= (COLOR_WHITE_L_THRESHOLD or 240):
            return "White", self.palette_bgr["White"], (x1, y1, x2, y2, cx, cy)
        if chroma <= (COLOR_GRAY_CHROMA or 10):
            return "Gray", self.palette_bgr["Gray"], (x1, y1, x2, y2, cx, cy)

        # Nearest palette by Euclidean distance in Lab
        best = None
        best_dist = float("inf")
        for name, pal_lab in self.palette_lab.items():
            dist = np.linalg.norm(mean_lab - pal_lab)
            if dist < best_dist:
                best_dist = dist
                best = name

        display_bgr = self.palette_bgr.get(best, (255, 255, 255))
        return best, display_bgr, (x1, y1, x2, y2, cx, cy)

    def process(self, frame):
        label, display_bgr, box = self._detect_label(frame)
        x1, y1, x2, y2, cx, cy = box

        self.buffer.append(label)
        try:
            modal_label = Counter(self.buffer).most_common(1)[0][0]
        except Exception:
            modal_label = label

        now = time.time()
        if modal_label != self.last_label:
            if now - self.last_spoken > (COLOR_TTS_COOLDOWN or 6.0):
                tts_queue.put(f"Color: {modal_label}")
                self.last_spoken = now
            self.last_label = modal_label

        if not HEADLESS_MODE:
            # Draw sample region and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), display_bgr, 2)
            cv2.circle(frame, (cx, cy), 3, (25, 25, 25), 2)

            text = modal_label
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            tx = x1
            ty = max(10, y1 - 10)
            cv2.rectangle(frame, (tx, ty - th - 6), (tx + tw + 6, ty + 4), (255, 255, 255), -1)
            cv2.putText(frame, text, (tx + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, display_bgr, 2, cv2.LINE_AA)

        return frame

    def summarize(self, frame):
        label, _, _ = self._detect_label(frame)
        return f"Dominant color: {label}"

    def reset(self):
        self.buffer.clear()
        self.last_label = None
        self.last_spoken = 0.0
