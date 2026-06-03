import argparse
import cv2
import os
import signal
import time
import threading
import queue
import sensors
import config
from flask import Flask, Response, jsonify
from config import (
    current_mode, mode_lock, tts_queue, active_mode_ref,
    MODE_NAMES, RESOLUTION, OCR_RESOLUTION, TTS_SHUTDOWN,
    THERMAL_ZONE_PATH, THERMAL_WARNING_THRESHOLD, THERMAL_CHECK_INTERVAL,
    STREAM_HOST, STREAM_PORT,
    llm_one_shot_queue, LOCAL_LLM_CHAT_MODE,GEMINI_CHAT_MODE, sensor_state_lock, latest_sensor_state
)
from camera import get_frame, release_camera, reconfigure_camera
from voice_control import start_voice_listener
from modes import CurrencyDetector, FaceRecognizer, GeminiSceneDescriber, LocalSceneDescriber, OCRProcessor, ObjectDetector, ColorRecognition, LightRecognition

parser = argparse.ArgumentParser(description="Smart glasses CV service")
parser.add_argument("--headless", action="store_true",
                    help="Run without display output and window rendering")
args, _ = parser.parse_known_args()
if args.headless:
    config.set_headless_mode(True)


# ==========================================
# CV STREAM SERVICE
# ==========================================
app = Flask(__name__)


def _frame_to_jpeg(frame):
    success, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes() if success else None


def mjpeg_generator():
    while True:
        frame = get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        jpeg = _frame_to_jpeg(frame)
        if jpeg is None:
            time.sleep(0.1)
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
        )


@app.route("/")
def index():
    return "Smart glasses CV service running. Use /viewer, /video, /snapshot.jpg, /health."


@app.route("/video")
def video():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/snapshot.jpg")
def snapshot():
    frame = get_frame()
    if frame is None:
        return Response("Camera not available", status=503)

    jpeg = _frame_to_jpeg(frame)
    if jpeg is None:
        return Response("Failed to encode snapshot", status=500)

    return Response(jpeg, mimetype="image/jpeg")


@app.route("/viewer")
def viewer():
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        html, body {
          margin: 0;
          padding: 0;
          background: black;
          width: 100%;
          height: 100%;
        }
        img {
          width: 100%;
          height: auto;
          display: block;
        }
      </style>
    </head>
    <body>
      <img src="/video" />
    </body>
    </html>
    """

@app.route("/metrics")
@app.route("/sensors")
def sensor_metrics():
    """Exposes all real-time structural sensor payloads directly on the primary CV stream port."""
    with sensor_state_lock:
        return jsonify(latest_sensor_state)

@app.route("/health")
def health():
    return {"status": "ok"}


# ==========================================
# PRELOAD MODELS
# ==========================================
processors = {}
failed_modes = {}  # mode_num -> error string

def preload_all():
    def _load():
        loaders = [
            (1, "Currency Detection", CurrencyDetector),
            (2, "Face Recognition", FaceRecognizer),
            (3, "OCR/Text Reading", OCRProcessor),
            (4, "Object Detection", ObjectDetector),
            (7, "Gemini Scene Description", GeminiSceneDescriber),
            (8, "Local LLM Scene Description", LocalSceneDescriber),
            (9, "Color Recognition", ColorRecognition),
            (10, "Light Recognition", LightRecognition),
        ]

        for mode_num, name, cls in loaders:
            print(f"[INFO] Loading {name}...")
            try:
                processors[mode_num] = cls()
                print(f"[INFO] ✓ {name} loaded successfully")
            except Exception as e:
                print(f"[ERROR] Failed to load {name}: {e}")
                failed_modes[mode_num] = str(e)

        tts_queue.put("All models loaded. Ready.")

    threading.Thread(target=_load, daemon=True).start()

# ==========================================
# THERMAL MONITORING (RPi)
# ==========================================
def _thermal_monitor():
    """Periodically check RPi CPU temperature and warn if throttling."""
    if not os.path.exists(THERMAL_ZONE_PATH):
        return  # Not on RPi or no thermal zone
    while True:
        try:
            with open(THERMAL_ZONE_PATH) as f:
                temp_c = int(f.read().strip()) / 1000
            if temp_c >= THERMAL_WARNING_THRESHOLD:
                tts_queue.put(f"Warning: device temperature is {int(temp_c)} degrees. Performance may be reduced.")
                print(f"[THERMAL] CPU temp: {temp_c:.1f}°C (above threshold)")
        except Exception:
            pass
        time.sleep(THERMAL_CHECK_INTERVAL)

# ==========================================
# MAIN LOOP
# ==========================================
def main():
# SIGTERM handler for clean systemd service shutdown
    def _signal_handler(sig, frame):
        with mode_lock:
            current_mode[0] = 0
    signal.signal(signal.SIGTERM, _signal_handler)

    active_mode = None

    # 1. Start the Unified Flask Server
    threading.Thread(target=lambda: app.run(host=STREAM_HOST, port=STREAM_PORT, threaded=True, use_reloader=False), daemon=True).start()
    print(f"[INFO] Unified server running at http://{STREAM_HOST}:{STREAM_PORT}")

    # 2. Fire up Voice Control Assistant
    start_voice_listener()
    preload_all()
    threading.Thread(target=_thermal_monitor, daemon=True).start()

    # =========================================================
    # 🔥 NEW: UNIFIED BACKGROUND HARDWARE TELEMETRY SUBSYSTEMS
    # =========================================================
    print("[INFO] Initializing hardware peripheral components...")
    sensors.init_firebase()

    # Start Bio-metric I2C interface safely
    try:
        sensors.heart_service.start()
        with sensor_state_lock:
            latest_sensor_state["system"]["heart_available"] = True
        threading.Thread(target=sensors.heart_uploader, daemon=True).start()
        print("[INFO] ✓ MAX30102 Heart Rate Engine Active.")
    except Exception as e:
        with sensor_state_lock:
            latest_sensor_state["system"]["heart_available"] = False
        print(f"[ERROR] Heart Rate hardware missing or blocked: {e}")

    # Kick off background asynchronous telemetry loops
    threading.Thread(target=sensors.gps_loop, daemon=True).start()
    threading.Thread(target=sensors.obstacle_loop, daemon=True).start()
    threading.Thread(target=sensors.system_uploader, daemon=True).start()
    print("[INFO] ✓ GPS, Ultrasonic, and Firebase state machines online.")
    tts_queue.put("Smart glasses starting. Loading models, please wait. Say help for all commands.")

    try:
        while True:
            with sensor_state_lock:
                latest_sensor_state["system"]["camera_running"] = (active_mode is not None)
                latest_sensor_state["system"]["camera_available"] = True
            _handle_one_shot_requests()

            with mode_lock:
                new_mode = current_mode[0]
                current_mode[0] = None

            if new_mode == 0:
                break

            if new_mode is not None and new_mode != active_mode:
                if new_mode in failed_modes:
                    tts_queue.put(f"{MODE_NAMES.get(new_mode, 'Mode')} failed to load and is unavailable.")
                elif new_mode == GEMINI_CHAT_MODE or new_mode == LOCAL_LLM_CHAT_MODE:
                    active_mode = new_mode
                    active_mode_ref[0] = active_mode
                    tts_queue.put(f"Switching to {MODE_NAMES[active_mode]}")
                    print(f"[MODE] Switched to: {MODE_NAMES[active_mode]}")
                elif new_mode in processors:
                            # Reset outgoing processor state (e.g. OCR recent_texts)
                    if active_mode in processors and hasattr(processors[active_mode], 'reset'):
                        processors[active_mode].reset()
                    # Reset the incoming processor to allow one-shot modes to re-run cleanly
                    if new_mode in processors and hasattr(processors[new_mode], 'reset'):
                        processors[new_mode].reset()

                    # Reconfigure camera for OCR (higher res) or back to default
                    target_res = OCR_RESOLUTION if new_mode == 3 else RESOLUTION
                    reconfigure_camera(target_res)
                    active_mode = new_mode
                    active_mode_ref[0] = active_mode
                    tts_queue.put(f"Switching to {MODE_NAMES[active_mode]}")
                    print(f"[MODE] Switched to: {MODE_NAMES[active_mode]}")
                else:
                    tts_queue.put("Still loading, please wait.")

            # Skip frame capture when idle in headless mode
            if active_mode is None and config.HEADLESS_MODE:
                time.sleep(0.1)
                continue

            frame = get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            if active_mode and active_mode in processors:
                try:
                    frame = processors[active_mode].process(frame)
                except Exception as e:
                    print(f"[ERROR] Mode {active_mode} processing failed: {e}")

                if getattr(processors[active_mode], "completed", False):
                    with mode_lock:
                        current_mode[0] = None
                    active_mode = None
                    active_mode_ref[0] = None

            if not config.HEADLESS_MODE:
                if active_mode:
                    cv2.putText(frame, f"Mode: {MODE_NAMES[active_mode].upper()}", (10,30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow("Smart Glasses", rgb)

                # Check if window is still open
                try:
                    if cv2.getWindowProperty("Smart Glasses", cv2.WND_PROP_VISIBLE) < 1:
                        break
                except:
                    break

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Throttle loop when headless to avoid pinning CPU at 100%
                time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Main loop error: {e}")
    finally:
        print("[INFO] Shutting down...")
        release_camera()
        try:
            import RPi.GPIO as GPIO
            GPIO.cleanup()
            print("[INFO] ✓ GPIO Pins cleaned up safely.")
        except Exception:
            pass
        if not config.HEADLESS_MODE:
            cv2.destroyAllWindows()
        tts_queue.put(TTS_SHUTDOWN)
        time.sleep(1)  # Give TTS time to finish


def _handle_one_shot_requests():
    try:
        request = llm_one_shot_queue.get_nowait()
    except queue.Empty:
        return

    mode_num = request.get("mode")
    response_queue = request.get("response_queue")

    if mode_num not in processors:
        try:
            response_queue.put("Requested mode is not ready.")
        except Exception:
            pass
        return

    frame = get_frame()
    if frame is None:
        try:
            response_queue.put("Camera not available.")
        except Exception:
            pass
        return

    try:
        summary = processors[mode_num].summarize(frame)
    except Exception as e:
        print(f"[ERROR] One-shot mode {mode_num} failed: {e}")
        summary = "Error running detection."

    try:
        response_queue.put(summary)
    except Exception:
        pass

if __name__ == "__main__":
    main()
