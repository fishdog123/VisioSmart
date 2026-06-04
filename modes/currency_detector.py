import cv2
import numpy as np
import time
from collections import deque

from config import CURRENCY_MODEL_PATH, MONEY, YOLO_CONF, HEADLESS_MODE, tts_queue, NO_DETECT_INTERVAL

# ==========================================
# MODE 1: CURRENCY DETECTOR
# ==========================================
class CurrencyDetector:
    def __init__(self):
        from ultralytics import YOLO
        print("[INFO] Loading YOLO model for currency...")
        self.model = YOLO(CURRENCY_MODEL_PATH, task="detect")
        self.money = MONEY
        self.bbox_colors = [(164,120,87),(68,148,228),(93,97,209),(178,182,133),(88,159,106),(96,202,231),(159,124,168),(169,162,241),(98,118,150),(172,176,184)]
        self.frame_rate_buffer = deque(maxlen=50)
        self.avg_frame_rate = 0
        self.last_spoken_time = 0
        self.cooldown = 5
        self.last_detect_time = time.time()
        self.last_no_detect_time = 0
        print("[INFO] Currency detector ready.")

    def process(self, frame):
        t_start = time.perf_counter()
        results = self.model(frame, verbose=False, conf=YOLO_CONF)
        detections = results[0].boxes
        total_money = 0
        counts = {}

        for i in range(len(detections)):
            conf = detections[i].conf.item()
            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)

            if len(xyxy.shape) == 1:
                xmin, ymin, xmax, ymax = xyxy
            else:
                xmin, ymin, xmax, ymax = xyxy

            classidx = int(detections[i].cls.item())

            if classidx in self.money:
                classname = self.money[classidx]
                value = int(classname.split()[0])
                total_money += value
                counts[classname] = counts.get(classname,0)+1
            else:
                classname = "Unknown"

            color = self.bbox_colors[classidx % len(self.bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, f"{classname}: {int(conf*100)}%", (xmin, ymin-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        now = time.time()
        if counts and (now-self.last_spoken_time) > self.cooldown:
            speech = [f"{cnt} {name}" for name, cnt in counts.items()]
            tts_queue.put("I see " + " and ".join(speech) + f". Total is {total_money} Pounds.")
            self.last_spoken_time = now
            self.last_detect_time = now

        # No-detection heartbeat
        if not counts and (now - self.last_detect_time) > NO_DETECT_INTERVAL \
                and (now - self.last_no_detect_time) > NO_DETECT_INTERVAL:
            tts_queue.put("No currency detected. Still scanning.")
            self.last_no_detect_time = now

        t_stop = time.perf_counter()
        fps = 1 / max(t_stop - t_start, 1e-6)
        self.frame_rate_buffer.append(fps)
        self.avg_frame_rate = np.mean(self.frame_rate_buffer)

        if not HEADLESS_MODE:
            cv2.putText(frame, f'Total: {total_money} Pounds', (10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(frame, f'FPS: {self.avg_frame_rate:.2f}', (10,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

        return frame

    def summarize(self, frame):
        """Run a one-shot inference and return a concise text summary."""
        results = self.model(frame, verbose=False, conf=YOLO_CONF)
        detections = results[0].boxes
        total_money = 0
        counts = {}

        for i in range(len(detections)):
            classidx = int(detections[i].cls.item())
            if classidx in self.money:
                classname = self.money[classidx]
                value = int(classname.split()[0])
                total_money += value
                counts[classname] = counts.get(classname, 0) + 1

        if not counts:
            return (0, "No currency detected.")

        speech = [f"{cnt}x {name}" for name, cnt in counts.items()]
        return (1, "Currency detected: " + ", ".join(speech) + f". Total value: {total_money} Pounds.")
