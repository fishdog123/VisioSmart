import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array


class EmotionRecognizer:
    """Helper to load a Keras emotion model and run per-face predictions.

    Usage:
        er = EmotionRecognizer(model_path)
        label, score = er.predict(frame, (x1, y1, x2, y2))
    """

    DEFAULT_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def __init__(self, model_path: str = None, input_size=(48, 48), labels=None, enabled=True):
        self.enabled = bool(enabled)
        self.model = None
        self.input_size = tuple(input_size)
        self.labels = labels or list(self.DEFAULT_LABELS)

        if not self.enabled or not model_path:
            self.enabled = False
            return

        if not os.path.exists(model_path):
            print(f"[WARN] Emotion model not found at {model_path}")
            self.enabled = False
            return

        try:
            from keras.models import load_model
            self.model = load_model(model_path)
            print(f"[INFO] Loaded emotion model from {model_path}")
        except Exception as e:
            print(f"[WARN] Failed to load emotion model: {e}")
            self.model = None
            self.enabled = False

    def predict(self, frame, bbox):
        """Predict emotion for the given BGR `frame` and integer bbox (x1,y1,x2,y2).

        Returns (label, score) where score is the softmax probability for the chosen label.
        """
        if not self.enabled or self.model is None:
            return ("Unknown", 0.0)

        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(frame.shape[1] - 1, int(x2)); y2 = min(frame.shape[0] - 1, int(y2))
        if x2 <= x1 or y2 <= y1:
            return ("Unknown", 0.0)

        roi_color = frame[y1:y2, x1:x2]
        try:
            roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.resize(roi_gray, self.input_size, interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            if roi.ndim == 2:
                roi = np.expand_dims(roi, -1)
            roi = np.expand_dims(roi, 0)

            preds = self.model.predict(roi)[0]
            idx = int(np.argmax(preds))
            label = self.labels[idx] if idx < len(self.labels) else "Unknown"
            score = float(preds[idx])
            return (label, score)
        except Exception as e:
            print(f"[WARN] Emotion prediction error: {e}")
            return ("Unknown", 0.0)
