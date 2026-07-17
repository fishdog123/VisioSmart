import os
import pickle
import cv2
import numpy as np
from imutils import paths
import insightface
from pathlib import Path
from config import BASE_DIR


DATASET_PATH = BASE_DIR / "face_detection" / "dataset"
OUTPUT_PATH = BASE_DIR / "face_detection" / "encodings_insightface_retina_small.pickle"
MODEL_NAME = "buffalo_sc"
DET_SIZE = (320, 320)
MIN_FACE_CONF = 0.70




def load_dataset(dataset_path=DATASET_PATH):
    if not dataset_path.exists():
        raise FileNotFoundError(f"[ERROR] Dataset folder not found: {dataset_path.absolute()}")

    print(f"[INFO] Scanning dataset: {dataset_path.absolute()}")

    image_paths = list(paths.list_images(str(dataset_path)))

    if not image_paths:
        raise ValueError(f"[ERROR] No images found inside '{dataset_path.name}'.")

    return image_paths

def embed_images(image_paths):
    known_encodings = []
    known_names = []

    for (i, image_path) in enumerate(image_paths):
        print(f"[INFO] Processing {i + 1}/{len(image_paths)}: {image_path}")
        name = os.path.basename(os.path.dirname(image_path))

        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] Skipping unreadable file: {image_path}")
            continue

        # Detect faces
        faces = model.get(image)
        if not faces:
            print(f"[WARN] No faces detected in {image_path}")
            continue

        # Choose the largest face (in case of multiple detections)
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # Skip low-confidence faces
        if hasattr(largest_face, "det_score") and largest_face.det_score < MIN_FACE_CONF:
            print(f"[WARN] Low confidence ({largest_face.det_score:.2f}) for {image_path}, skipping.")
            continue

        embedding = largest_face.normed_embedding   # normalize it here instead of later
        known_encodings.append(embedding)
        known_names.append(name)

    return (known_encodings, known_names)


def save(encodings, names, output_path=OUTPUT_PATH):
    if not encodings:
        print("[WARN] No encodings generated. Nothing saved.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving {len(encodings)} face embeddings to '{output_path.absolute()}'...")

    data = {"encodings": encodings, "names": names}
    try:
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        print("[INFO] Training complete! Database saved successfully.")
    except Exception as e:
        print(f"[ERROR] Could not save file: {e}")


def main():
    global model

    model = insightface.app.FaceAnalysis(name=MODEL_NAME)
    model.prepare(ctx_id=-1, det_size=DET_SIZE)            # ctx -1 for CPU  0 for GPU

    print("[INFO] Model initialized successfully.")

    image_paths = load_dataset(DATASET_PATH)
    encodings, names = embed_images(image_paths)
    save(encodings, names, OUTPUT_PATH)


if __name__ == "__main__":
    main()
