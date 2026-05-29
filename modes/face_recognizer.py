import cv2
import numpy as np
import os
import time
from collections import defaultdict
from config import (
    FACE_DB_PATH,
    FACE_DETECT_INTERVAL,
    FACE_THRESHOLD,
    HEADLESS_MODE,
    tts_queue,
    NO_PERSON_GRACE,
    PERSON_TTL,
    ANNOUNCE_EVERY,
    GREET_COOLDOWN,
    EMOTION_ENABLED,
    EMOTION_MODEL_PATH,
    EMOTION_INPUT_SIZE,
    EMOTION_CONFIDENCE_THRESHOLD,
    EMOTION_COOLDOWN,
    EMOTION_DETECT_INTERVAL,
)


class FaceRecognizer:
    def __init__(self):
        import insightface
        import pickle

        print("[INFO] Loading InsightFace model...")
        self.app = insightface.app.FaceAnalysis(name="buffalo_sc")
        self.app.prepare(ctx_id=-1, det_size=(160, 160), det_thresh=0.5)

        self.person_db = {}
        if os.path.exists(FACE_DB_PATH):
            with open(FACE_DB_PATH, "rb") as f:
                data = pickle.load(f)

            raw_embeddings = defaultdict(list)
            for emb, nm in zip(data.get("encodings", []), data.get("names", [])):
                nm = nm.strip().lower()
                emb = np.asarray(emb, dtype=np.float32)
                emb = emb / np.linalg.norm(emb)
                raw_embeddings[nm].append(emb)

            for nm, embs in raw_embeddings.items():
                mean_emb = np.mean(embs, axis=0)
                self.person_db[nm] = mean_emb / np.linalg.norm(mean_emb)

        # ---------------- UX state ----------------

        self.frame_count = 0
        self.last_faces = []

        self.last_face_seen_time = 0.0

        # name -> last_seen_time
        self.active_people = {}
        self.person_positions = {}          # name -> "on the left" / "in the center" / "on the right"
        self.greeted_times = {}             # name -> last_greeted_time (arrival cooldown)

        self.unknown_count = 0
        self.unknown_positions = []         # list of position strings for current unknowns
        self.last_unknown_seen_time = 0.0
        self._recent_unknown_counts = []   # rolling window for smoothing
        self.UNKNOWN_SMOOTH_FRAMES = 3     # require consistency over N detection cycles

        self.last_announced_state = (frozenset(), 0)
        self.last_announce_time = time.time()

        self.no_person_announced = False

        # Emotion state
        self.person_emotions = {}    # name -> {'label', 'score', 'ts'}
        self.unknown_emotions = []   # list of (position, label, score)
        self.last_emotion_announce = {}  # name -> ts

        # Load emotion recognizer (modular helper)
        if EMOTION_ENABLED:
            from .emotion_recognizer import EmotionRecognizer
            try:
                self.emotion = EmotionRecognizer(model_path=EMOTION_MODEL_PATH, input_size=EMOTION_INPUT_SIZE)
            except Exception as e:
                print(f"[WARN] Failed to initialize EmotionRecognizer: {e}")
                self.emotion = None
        else:
            self.emotion = None

        print("[INFO] Face recognition ready.")

    def reset(self):
        """Clear all UX state on mode switch."""
        self.frame_count = 0
        self.last_faces = []
        self.last_face_seen_time = 0.0
        self.active_people.clear()
        self.person_positions.clear()
        self.greeted_times.clear()
        self.unknown_count = 0
        self.unknown_positions = []
        self.last_unknown_seen_time = 0.0
        self._recent_unknown_counts.clear()
        self.last_announced_state = (frozenset(), 0)
        self.last_announce_time = time.time()
        self.no_person_announced = False
        # Clear emotion-related UX state
        try:
            self.person_emotions.clear()
        except Exception:
            self.person_emotions = {}
        self.unknown_emotions = []
        try:
            self.last_emotion_announce.clear()
        except Exception:
            self.last_emotion_announce = {}

    def _get_position(self, x1, x2, frame_width):
        """Return spatial zone string based on horizontal position."""
        center_x = (x1 + x2) / 2
        third = frame_width / 3
        if center_x < third:
            return "on the left"
        elif center_x < 2 * third:
            return "in the center"
        else:
            return "on the right"

    def recognize(self, embedding):
        if not self.person_db:
            return "Unknown"

        best_name, best_sim = "Unknown", -1.0
        for nm, mean_emb in self.person_db.items():
            sim = float(np.dot(mean_emb, embedding))
            if sim > best_sim:
                best_sim, best_name = sim, nm

        return best_name if best_sim >= FACE_THRESHOLD else "Unknown"

    def process(self, frame):

        self.frame_count += 1
        now = time.time()
        frame_width = frame.shape[1]

        # -------------------------------
        # Run detector every N frames
        # -------------------------------
        if self.frame_count % FACE_DETECT_INTERVAL == 0:

            faces = self.app.get(frame)

            self.last_faces.clear()

            seen_names_this_frame = set()
            unknown_count = 0
            frame_unknown_positions = []
            frame_unknown_emotions = []

            for face in faces:
                name = self.recognize(face.normed_embedding) \
                    if face.normed_embedding is not None else "Unknown"

                x1, y1, x2, y2 = face.bbox.astype(int)
                position = self._get_position(x1, x2, frame_width)

                # predict emotion for this face (if available)
                emotion_label = None
                emotion_score = 0.0
                if self.emotion and getattr(self.emotion, 'enabled', True):
                    try:
                        emotion_label, emotion_score = self.emotion.predict(frame, (x1, y1, x2, y2))
                    except Exception as e:
                        emotion_label, emotion_score = (None, 0.0)

                if not HEADLESS_MODE:
                    self.last_faces.append((x1, y1, x2, y2, name, emotion_label, emotion_score))

                # update global "face seen" heartbeat
                self.last_face_seen_time = now

                if name != "Unknown":
                    was_new = name not in self.active_people
                    self.active_people[name] = now
                    self.person_positions[name] = position
                    seen_names_this_frame.add(name)

                    # store last-seen emotion for this person
                    self.person_emotions[name] = {'label': emotion_label, 'score': emotion_score, 'ts': now}

                    # Instant arrival greeting
                    if was_new:
                        last_greet = self.greeted_times.get(name, 0)
                        if now - last_greet > GREET_COOLDOWN:
                            greeting = f"{name.title()} is here, {position}"
                            last_emotion = self.last_emotion_announce.get(name, 0)
                            if (
                                emotion_label
                                and emotion_score >= EMOTION_CONFIDENCE_THRESHOLD
                                and now - last_emotion > EMOTION_COOLDOWN
                            ):
                                greeting = f"{greeting}, looking {emotion_label.lower()}"
                                self.last_emotion_announce[name] = now
                            tts_queue.put(greeting)
                            self.greeted_times[name] = now
                else:
                    unknown_count += 1
                    frame_unknown_positions.append(position)
                    if emotion_label:
                        frame_unknown_emotions.append((position, emotion_label, emotion_score))

            # Update unknown face tracking (smoothed)
            self._recent_unknown_counts.append(unknown_count)
            if len(self._recent_unknown_counts) > self.UNKNOWN_SMOOTH_FRAMES:
                self._recent_unknown_counts.pop(0)

            stable_unknown = min(self._recent_unknown_counts)
            if stable_unknown > 0:
                self.unknown_count = stable_unknown
                self.unknown_positions = frame_unknown_positions
                self.unknown_emotions = frame_unknown_emotions
                self.last_unknown_seen_time = now
            elif unknown_count == 0:
                # Only clear immediately when this frame sees zero unknowns
                self.unknown_count = 0
                self.unknown_positions = []
                self.unknown_emotions = []

            # -------------------------------
            # Prune people who left
            # -------------------------------
            to_remove = []
            for name, t in self.active_people.items():
                if now - t > PERSON_TTL:
                    to_remove.append(name)

            if to_remove:
                departed = " and ".join(n.title() for n in sorted(to_remove))
                tts_queue.put(f"{departed} left")
                for n in to_remove:
                    del self.active_people[n]
                    self.person_positions.pop(n, None)

            # Expire unknowns after PERSON_TTL
            if now - self.last_unknown_seen_time > PERSON_TTL:
                self.unknown_count = 0
                self.unknown_positions = []

        # -------------------------------
        # 15s grouped announcement
        # -------------------------------
        if now - self.last_announce_time >= ANNOUNCE_EVERY:

            current_set = frozenset(self.active_people.keys())
            current_state = (current_set, self.unknown_count)

            if current_state != self.last_announced_state:

                parts = []
                # Known people with positions
                if current_set:
                    named_parts = []
                    for n in sorted(current_set):
                        pos = self.person_positions.get(n, "")
                        em = self.person_emotions.get(n, {})
                        label = em.get('label') if isinstance(em, dict) else None
                        score = em.get('score', 0.0) if isinstance(em, dict) else 0.0
                        if label and score >= EMOTION_CONFIDENCE_THRESHOLD:
                            if pos:
                                named_parts.append(f"{n.title()} ({label.lower()}) {pos}")
                            else:
                                named_parts.append(f"{n.title()} ({label.lower()})")
                        else:
                            named_parts.append(f"{n.title()} {pos}" if pos else n.title())
                    parts.append(", ".join(named_parts))

                # Unknown people with positions
                if self.unknown_count > 0:
                    if self.unknown_positions:
                        # Group unknowns by position and include common emotion if available
                        from collections import Counter
                        pos_counts = Counter(self.unknown_positions)
                        unknown_parts = []
                        for pos, cnt in sorted(pos_counts.items()):
                            # find most common emotion for this position
                            labels = [e[1] for e in self.unknown_emotions if e[0] == pos]
                            common_label = None
                            if labels:
                                from collections import Counter as C2
                                common_label = C2(labels).most_common(1)[0][0]
                            if cnt == 1:
                                if common_label:
                                    unknown_parts.append(f"1 unknown person ({common_label.lower()}) {pos}")
                                else:
                                    unknown_parts.append(f"1 unknown person {pos}")
                            else:
                                if common_label:
                                    unknown_parts.append(f"{cnt} unknown people ({common_label.lower()}) {pos}")
                                else:
                                    unknown_parts.append(f"{cnt} unknown people {pos}")
                        parts.append(", ".join(unknown_parts))
                    else:
                        if self.unknown_count == 1:
                            parts.append("1 unknown person")
                        else:
                            parts.append(f"{self.unknown_count} unknown people")

                if parts:
                    tts_queue.put(f"I see {' and '.join(parts)}")
                    self.no_person_announced = False

                self.last_announced_state = current_state

            self.last_announce_time = now

        # -------------------------------
        # No person detected logic
        # -------------------------------
        if len(self.active_people) == 0 and self.unknown_count == 0:

            if (
                self.last_face_seen_time > 0 and
                now - self.last_face_seen_time > NO_PERSON_GRACE and
                not self.no_person_announced
            ):
                tts_queue.put("No person detected")
                self.no_person_announced = True

        else:
            # reset when someone is present again
            self.no_person_announced = False

        # -------------------------------
        # Display
        # -------------------------------
        if not HEADLESS_MODE:
            for x1, y1, x2, y2, name, emotion, emotion_score in self.last_faces:
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text = name.title()
                if emotion and emotion != "Unknown":
                    label_text = f"{label_text} ({emotion} {emotion_score:.2f})"
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame

    def summarize(self, frame):
        """Run a one-shot face recognition and return a concise text summary."""
        faces = self.app.get(frame)
        if not faces:
            return "No person detected."

        frame_width = frame.shape[1]
        named = []  # list of (name, position, emotion_label_or_None)
        unknown_positions = []
        unknown_emotions = []  # list of (position, label)

        for face in faces:
            name = self.recognize(face.normed_embedding) \
                if face.normed_embedding is not None else "Unknown"
            x1, y1, x2, y2 = face.bbox.astype(int)
            position = self._get_position(x1, x2, frame_width)

            # get emotion for this face if available
            emotion_label = None
            emotion_score = 0.0
            if self.emotion and getattr(self.emotion, 'enabled', True):
                try:
                    emotion_label, emotion_score = self.emotion.predict(frame, (x1, y1, x2, y2))
                except Exception:
                    emotion_label, emotion_score = (None, 0.0)

            if name != "Unknown":
                if emotion_label and emotion_score >= EMOTION_CONFIDENCE_THRESHOLD:
                    named.append((name, position, emotion_label))
                else:
                    named.append((name, position, None))
            else:
                unknown_positions.append(position)
                if emotion_label and emotion_score >= EMOTION_CONFIDENCE_THRESHOLD:
                    unknown_emotions.append((position, emotion_label))

        parts = []
        if named:
            named_parts = []
            for name, position, label in named:
                if label:
                    named_parts.append(f"{name.title()} ({label.lower()}) {position}")
                else:
                    named_parts.append(f"{name.title()} {position}")
            parts.append(", ".join(named_parts))

        if unknown_positions:
            if len(unknown_positions) == 1:
                if unknown_emotions:
                    parts.append(f"1 unknown person ({unknown_emotions[0][1].lower()}) {unknown_positions[0]}")
                else:
                    parts.append(f"1 unknown person {unknown_positions[0]}")
            else:
                if unknown_emotions:
                    labels = [e[1] for e in unknown_emotions]
                    from collections import Counter
                    common_label = Counter(labels).most_common(1)[0][0]
                    parts.append(f"{len(unknown_positions)} unknown people ({common_label.lower()})")
                else:
                    parts.append(f"{len(unknown_positions)} unknown people")

        if parts:
            return "I see " + " and ".join(parts)
        return "No person detected."
