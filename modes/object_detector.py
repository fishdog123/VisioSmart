import cv2
import numpy as np
import os
import time
from collections import deque
from config import OBJECT_MODEL_PATH, BASE_DIR, YOLO_CONF, HEADLESS_MODE, tts_queue, NO_DETECT_INTERVAL


class ObjectDetector:
    def __init__(self):
        from ultralytics import YOLO
        print("[INFO] Loading YOLO model for object detection...")

        if not os.path.exists(OBJECT_MODEL_PATH):
            print(f"[WARNING] Model path not found: {OBJECT_MODEL_PATH}")
            base_obj_dir = BASE_DIR / "object_detection"
            if not base_obj_dir.exists():
                raise FileNotFoundError(f"Object detection directory not found: {base_obj_dir}")
            for item in base_obj_dir.iterdir():
                if item.is_dir() and 'ncnn' in item.name.lower():
                    self.model_path = str(item)
                    print(f"[INFO] Found alternative model path: {self.model_path}")
                    break
            else:
                raise FileNotFoundError(f"Object detection model not found in {base_obj_dir}")
        else:
            self.model_path = OBJECT_MODEL_PATH

        print(f"[INFO] Loading model from: {self.model_path}")
        self.model = YOLO(self.model_path, task="detect")

        # Use YOLO model's built-in class names
        self.class_names = self.model.names

        self.irregular_plurals = {
            "person": "people", "sheep": "sheep", "mouse": "mice",
            "knife": "knives", "child": "children", "bus": "buses",
        }

        self.bbox_colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),
                            (255,0,255),(0,255,255),(128,128,0),(128,0,128),
                            (0,128,128),(64,64,64)]
        self.frame_rate_buffer = deque(maxlen=50)
        self.avg_frame_rate = 0
        self.last_spoken_time = 0
        self.cooldown = 10
        self.last_detect_time = time.time()
        self.last_no_detect_time = 0
        print("[INFO] Object detector ready.")

    def _get_position(self, xmin, xmax, frame_width):
        """Determine horizontal position: left, center, or right."""
        center_x = (xmin + xmax) / 2
        third = frame_width / 3
        if center_x < third:
            return "on the left"
        elif center_x < 2 * third:
            return "in the center"
        else:
            return "on the right"

    def process(self, frame):
        t_start = time.perf_counter()
        frame_width = frame.shape[1]

        results = self.model(frame, verbose=False, conf=YOLO_CONF)
        detections = results[0].boxes
        object_details = []

        for i in range(len(detections)):
            conf = detections[i].conf.item()

            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
            if len(xyxy.shape) == 1:  # Single detection
                xmin, ymin, xmax, ymax = xyxy
            else:
                continue

            classidx = int(detections[i].cls.item())
            classname = self.class_names.get(classidx, f"object_{classidx}")
            position = self._get_position(xmin, xmax, frame_width)
            object_details.append((classname, position))

            color = self.bbox_colors[classidx % len(self.bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            label = f"{classname}: {int(conf*100)}%"
            cv2.putText(frame, label, (xmin, ymin-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        now = time.time()
        if object_details and (now - self.last_spoken_time) > self.cooldown:
            grouped = {}
            for classname, position in object_details:
                key = (classname, position)
                grouped[key] = grouped.get(key, 0) + 1

            speech_parts = []
            for (name, position), cnt in grouped.items():
                if cnt == 1:
                    article = "an" if name[0] in 'aeiou' else "a"
                    speech_parts.append(f"{article} {name} {position}")
                else:
                    plural = self.irregular_plurals.get(name)
                    if plural is None:
                        if name.endswith(('s', 'sh', 'ch', 'x', 'z')):
                            plural = name + "es"
                        else:
                            plural = name + "s"
                    speech_parts.append(f"{cnt} {plural} {position}")

            if speech_parts:
                if len(speech_parts) == 1:
                    speech_text = f"I see {speech_parts[0]}"
                elif len(speech_parts) == 2:
                    speech_text = f"I see {speech_parts[0]} and {speech_parts[1]}"
                else:
                    speech_text = f"I see {', '.join(speech_parts[:-1])}, and {speech_parts[-1]}"

                tts_queue.put(speech_text)
                self.last_spoken_time = now

        # No-detection heartbeat
        if not object_details and (now - self.last_detect_time) > NO_DETECT_INTERVAL \
                and (now - self.last_no_detect_time) > NO_DETECT_INTERVAL:
            tts_queue.put("No objects detected. Still scanning.")
            self.last_no_detect_time = now
        elif object_details:
            self.last_detect_time = now

        # Calculate FPS
        t_stop = time.perf_counter()
        fps = 1 / max(t_stop - t_start, 1e-6)
        self.frame_rate_buffer.append(fps)
        self.avg_frame_rate = np.mean(self.frame_rate_buffer)

        if not HEADLESS_MODE:
            counts = {}
            for name, _ in object_details:
                counts[name] = counts.get(name, 0) + 1
            y_offset = 20
            for name, cnt in counts.items():
                cv2.putText(frame, f"{cnt} {name}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y_offset += 20

            cv2.putText(frame, f'FPS: {self.avg_frame_rate:.2f}', (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        return frame

    def summarize(self, frame):
        frame_width = frame.shape[1]
        results = self.model(frame, verbose=False, conf=YOLO_CONF)
        detections = results[0].boxes

        grouped = {}
        for i in range(len(detections)):
            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
            if len(xyxy.shape) != 1:
                continue
            xmin, ymin, xmax, ymax = xyxy
            classidx = int(detections[i].cls.item())
            classname = self.class_names.get(classidx, f"object_{classidx}")
            position = self._get_position(xmin, xmax, frame_width)
            key = (classname, position)
            grouped[key] = grouped.get(key, 0) + 1

        if not grouped:
            return (0, "No objects detected.")

        data_parts = [f"{cnt}x {name} ({position})" for (name, position), cnt in grouped.items()]
        return (1, "Detected objects: " + ", ".join(data_parts))
