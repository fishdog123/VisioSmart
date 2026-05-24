import cv2
import numpy as np
import time
from collections import deque

from config import OCR_BASE_DIR, OCR_INTERVAL, OCR_MIN_CONFIDENCE, SHOW_DISPLAY, tts_queue, NO_DETECT_INTERVAL

# ==========================================
# MODE 3: OCR
# ==========================================
class OCRProcessor:
    def __init__(self):
        try:
            from onnxocr.onnx_paddleocr import ONNXPaddleOcr
            print("[INFO] Loading OCR engine...")
            self.ocr = ONNXPaddleOcr(
                det_model_dir=str(OCR_BASE_DIR / "paddleocr-onnx/detection/v3/det.onnx"),
                rec_model_dir=str(OCR_BASE_DIR / "paddleocr-onnx/languages/english/rec.onnx"),
                rec_char_dict_path=str(OCR_BASE_DIR / "paddleocr-onnx/languages/english/dict.txt"),
                use_gpu=False
            )
        except ImportError:
            print("[WARNING] ONNXPaddleOcr not available, using EasyOCR fallback")
            import easyocr
            self.ocr = easyocr.Reader(['en'])
            self.use_easyocr = True
        else:
            self.use_easyocr = False

        self.prev_ocr_time = 0
        self.recent_texts = deque(maxlen=20)
        self.last_boxes = []
        self.last_detect_time = time.time()
        self.last_no_detect_time = 0

    def reset(self):
        """Clear state when leaving OCR mode so text is re-announced on return."""
        self.recent_texts.clear()
        self.last_boxes = []
        self.last_detect_time = time.time()
        self.last_no_detect_time = 0

    def _sort_by_position(self, items):
        """Sort OCR results top-to-bottom, then left-to-right for natural reading order."""
        return sorted(items, key=lambda item: (item[0][0][1], item[0][0][0]))

    def process(self, frame):
        now = time.time()
        if now - self.prev_ocr_time >= OCR_INTERVAL:
            self.prev_ocr_time = now
            self.last_boxes = []
            raw_results = []

            if self.use_easyocr:
                results = self.ocr.readtext(frame)
                for (bbox, text, score) in results:
                    if score < OCR_MIN_CONFIDENCE or not text.strip():
                        continue
                    text = text.strip()
                    pts = [(int(p[0]), int(p[1])) for p in bbox]
                    raw_results.append((pts, text))
            else:
                results = self.ocr.ocr(frame, cls=False)
                for line in results:
                    for item in line:
                        box, (text, score) = item
                        text = text.strip()
                        if score < OCR_MIN_CONFIDENCE or not text:
                            continue
                        pts = [(int(p[0]), int(p[1])) for p in box]
                        raw_results.append((pts, text))

            # Sort by reading order (top-to-bottom, left-to-right)
            raw_results = self._sort_by_position(raw_results)

            # Collect new texts to speak as one combined message
            new_texts = []
            for pts, text in raw_results:
                if SHOW_DISPLAY:
                    self.last_boxes.append((pts, text))
                key = text.lower()
                if key not in self.recent_texts:
                    self.recent_texts.append(key)
                    new_texts.append(text)

            # Speak all new text fragments as one combined sentence
            if new_texts:
                combined = ". ".join(new_texts)
                tts_queue.put(combined)
                self.last_detect_time = now

            # No-detection heartbeat
            elif (now - self.last_detect_time) > NO_DETECT_INTERVAL \
                    and (now - self.last_no_detect_time) > NO_DETECT_INTERVAL:
                tts_queue.put("No text detected. Still scanning.")
                self.last_no_detect_time = now

        if SHOW_DISPLAY:
            for pts, text in self.last_boxes:
                cv2.polylines(frame, [np.array(pts)], True, (0,0,255), 2)
                cv2.putText(frame, text, (pts[0][0], pts[0][1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        return frame

    def summarize(self, frame):
        """Run a one-shot OCR pass and return a concise text summary."""
        raw_results = []

        if self.use_easyocr:
            results = self.ocr.readtext(frame)
            for (bbox, text, score) in results:
                if score < OCR_MIN_CONFIDENCE or not text.strip():
                    continue
                text = text.strip()
                pts = [(int(p[0]), int(p[1])) for p in bbox]
                raw_results.append((pts, text))
        else:
            results = self.ocr.ocr(frame, cls=False)
            for line in results:
                for item in line:
                    box, (text, score) = item
                    text = text.strip()
                    if score < OCR_MIN_CONFIDENCE or not text:
                        continue
                    pts = [(int(p[0]), int(p[1])) for p in box]
                    raw_results.append((pts, text))

        if not raw_results:
            return "No text detected."

        raw_results = self._sort_by_position(raw_results)
        combined = ". ".join([text for _, text in raw_results])
        return combined
