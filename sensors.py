# sensors.py
import time
import threading
import serial
import numpy as np
from collections import deque
from datetime import datetime, timezone
import RPi.GPIO as GPIO
from smbus2 import SMBus
import firebase_admin
from firebase_admin import credentials, db, firestore
from image_capture import create_folder
import requests
from create_face_embedding import main as create_face_embedding_main

from config import (
    FIREBASE_DB_URL, FIREBASE_KEY_PATH, DEVICE_ID,
    GPS_SERIAL_PORT, GPS_BAUD_RATE, TRIG_PIN, ECHO_PIN, BUZZER_PIN,
    OBSTACLE_THRESHOLD_CM, I2C_BUS, MAX30102_ADDR,
    latest_sensor_state, sensor_state_lock, BASE_DIR
)
import cv2

# =========================================================
# INITIALIZATION & FIREBASE REFERENCING
# =========================================================
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_KEY_PATH)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})

    db = firestore.client()
    people_ref = (
        db.collection("persons")
        .document("gp6meBa7X3XS6vYAlmzId39eJVH2")
        .collection("people")
    )

    known_people = {}
    for doc in people_ref.stream():
        data = doc.to_dict()
        known_people[data.get("name", "Unknown")] =  data.get("photoUrls", [])

    for name, urls in known_people.items():
        for url in urls:
            response = requests.get(url)
            if response.status_code == 200:
                image_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
                img_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                create_folder(BASE_DIR / "face_detection" / "dataset" / name)
                filename = url.split("/")[-1].split("?")[0]
                filepath = BASE_DIR / "face_detection" / "dataset" / name / filename
                cv2.imwrite(str(filepath), img_bgr)
                print(f"Downloaded {name}'s photo: {filepath}")
    create_face_embedding_main()


    with sensor_state_lock:
        latest_sensor_state["system"]["firebase_ok"] = True

def ref_gps(): return db.reference("gps/latest")
def ref_heart(): return db.reference(f"devices/{DEVICE_ID}/heartRate")
def ref_obstacle(): return db.reference(f"devices/{DEVICE_ID}/obstacle")
def ref_system(): return db.reference(f"devices/{DEVICE_ID}/system")

# =========================================================
# GPS MODULE INTERFACE
# =========================================================
def parse_lat(lat_str, ns):
    if not lat_str: return None
    raw = float(lat_str)
    deg = int(raw / 100)
    return -(deg + (raw - deg * 100) / 60) if ns == "S" else (deg + (raw - deg * 100) / 60)

def parse_lon(lon_str, ew):
    if not lon_str: return None
    raw = float(lon_str)
    deg = int(raw / 100)
    return -(deg + (raw - deg * 100) / 60) if ew == "W" else (deg + (raw - deg * 100) / 60)

def gps_loop():
    while True:
        try:
            with serial.Serial(GPS_SERIAL_PORT, GPS_BAUD_RATE, timeout=1) as ser:
                while True:
                    line = ser.readline().decode("ascii", errors="ignore").strip()
                    if not (line.startswith("$GPRMC") or line.startswith("$GNRMC")):
                        continue
                    try:
                        f = line[1:].split("*")[0].split(",")
                        if f[2] != "A": continue

                        lat = parse_lat(f[3], f[4])
                        lon = parse_lon(f[5], f[6])
                        payload = {
                            "latitude": round(lat, 7) if lat is not None else None,
                            "longitude": round(lon, 7) if lon is not None else None,
                            "speed_knots": float(f[7]) if f[7] else None,
                            "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
                        }
                        ref_gps().set(payload)
                        with sensor_state_lock:
                            latest_sensor_state["gps"] = {**payload, "status": "ok"}
                        time.sleep(1)
                    except Exception:
                        with sensor_state_lock:
                            latest_sensor_state["gps"]["status"] = "parse_error"
        except Exception:
            with sensor_state_lock:
                latest_sensor_state["gps"]["status"] = "serial_error"
            time.sleep(2)

# =========================================================
# ULTRASONIC RANGEFINDER & ALERTS
# =========================================================
def init_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.output(TRIG_PIN, False)
    GPIO.output(BUZZER_PIN, False)

def read_distance_cm():
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    # 1. Wait for Echo pin to go HIGH (Signal Start)
    # Giving it max 20ms to respond
    start_timeout = time.time() + 0.02
    pulse_start = time.time()
    while GPIO.input(ECHO_PIN) == 0:
        pulse_start = time.time()
        if time.time() > start_timeout:
            return None

    # 2. Wait for Echo pin to go LOW (Signal End)
    # Giving it max 0.03s (~30ms is roughly 5 meters max range tracking)
    end_timeout = time.time() + 0.03
    pulse_end = time.time()
    while GPIO.input(ECHO_PIN) == 1:
        pulse_end = time.time()
        if time.time() > end_timeout:
            return None

    return round((pulse_end - pulse_start) * 17150, 2)

def obstacle_loop():
    init_gpio()
    while True:
        try:
            distance = read_distance_cm()
            now_ts = int(time.time())
            alert = distance is not None and distance <= OBSTACLE_THRESHOLD_CM

            with sensor_state_lock:
                latest_sensor_state["obstacle"] = {
                    "distance_cm": distance, "threshold_cm": OBSTACLE_THRESHOLD_CM,
                    "alert": alert, "ts": now_ts, "status": "alert" if alert else "clear"
                }
            ref_obstacle().set({"distance_cm": distance, "threshold_cm": OBSTACLE_THRESHOLD_CM, "alert": alert, "timestamp": now_ts})

            if alert:
                GPIO.output(BUZZER_PIN, True)
                time.sleep(0.08)
                GPIO.output(BUZZER_PIN, False)
                time.sleep(0.08)
            else:
                time.sleep(0.15)
        except Exception:
            with sensor_state_lock:
                latest_sensor_state["obstacle"]["status"] = "error"
            time.sleep(0.3)

# =========================================================
# BIO-METRIC CONTROLLER (MAX30102)
# =========================================================
REG_INTR_ENABLE_1 = 0x02
REG_FIFO_WR_PTR, REG_OVF_COUNTER, REG_FIFO_RD_PTR = 0x04, 0x05, 0x06
REG_FIFO_DATA, REG_FIFO_CONFIG, REG_MODE_CONFIG, REG_SPO2_CONFIG = 0x07, 0x08, 0x09, 0x0A
REG_LED1_PA, REG_LED2_PA, REG_PILOT_PA, REG_PART_ID = 0x0C, 0x0D, 0x10, 0xFF

class MAX30102:
    def __init__(self):
        self.address = MAX30102_ADDR
        self.bus = None  # Do NOT call SMBus(I2C_BUS) here anymore

    def write_reg(self, reg, val): self.bus.write_byte_data(self.address, reg, val & 0xFF)
    def read_reg(self, reg): return self.bus.read_byte_data(self.address, reg)
    def read_regs(self, reg, length): return self.bus.read_i2c_block_data(self.address, reg, length)

    def init(self):
        # Open the physical hardware bus safely when init is executed at runtime
        self.bus = SMBus(I2C_BUS)

        if self.read_reg(REG_PART_ID) != 0x15:
            raise RuntimeError("MAX30102 missing")

        self.write_reg(REG_MODE_CONFIG, 0x40) # Reset
        time.sleep(0.05)
        self.write_reg(REG_INTR_ENABLE_1, 0x40)
        self.write_reg(REG_FIFO_CONFIG, (0b010 << 5) | (1 << 4) | 0x0F)
        self.write_reg(REG_SPO2_CONFIG, (0b01 << 5) | (0b011 << 2) | 0b11)
        self.write_reg(REG_LED1_PA, 0x24)
        self.write_reg(REG_LED2_PA, 0x24)
        self.write_reg(REG_PILOT_PA, 0x10)
        self.write_reg(REG_MODE_CONFIG, 0x03)
        self.write_reg(REG_FIFO_WR_PTR, 0x00)
        self.write_reg(REG_OVF_COUNTER, 0x00)
        self.write_reg(REG_FIFO_RD_PTR, 0x00)

    def read_fifo_samples(self):
        wr, rd = self.read_reg(REG_FIFO_WR_PTR), self.read_reg(REG_FIFO_RD_PTR)
        n = (wr - rd) & 0x1F
        if n == 0: return []
        raw = self.read_regs(REG_FIFO_DATA, n * 6)
        return [(((raw[i*6] & 0x03) << 16) | (raw[i*6+1] << 8) | raw[i*6+2],
                 ((raw[i*6+3] & 0x03) << 16) | (raw[i*6+4] << 8) | raw[i*6+5]) for i in range(n)]

def compute_bpm_from_ir(ir_values, fs=100):
    if len(ir_values) < fs * 3:        # Reduced from 5s to 3s for faster response
        return None

    x = np.array(ir_values, dtype=np.float64)

    # Check signal strength
    dc = np.mean(x[-fs * 2:])
    if dc < 5000:                      # No finger
        return None

    # Remove baseline (DC component)
    ma_win = int(fs * 0.8)
    baseline = np.convolve(x, np.ones(ma_win) / ma_win, mode="same")
    y = x - baseline

    # Smooth signal
    smooth_win = int(fs * 0.15)
    y = np.convolve(y, np.ones(smooth_win) / smooth_win, mode="same")

    # Use recent data
    recent = y[-fs * 4:]
    if len(recent) < fs * 2:
        return None

    # Dynamic threshold
    amp = np.percentile(np.abs(recent), 85)
    if amp < 8:                        # Increased minimum amplitude
        return None

    thresh = 0.38 * amp
    min_dist = int(fs * 0.45)          # Slightly increased to reduce false peaks

    peaks = []
    last_peak = -10**9

    for i in range(1, len(recent) - 1):
        if recent[i] > thresh and recent[i] > recent[i-1] and recent[i] > recent[i+1]:
            if i - last_peak >= min_dist:
                peaks.append(i)
                last_peak = i

    if len(peaks) < 2:
        return None

    intervals = np.diff(peaks) / fs
    # Filter realistic heart beat intervals (0.45s ~ 1.6s → 37~133 BPM)
    intervals = intervals[(intervals > 0.45) & (intervals < 1.6)]

    if len(intervals) < 2:
        return None

    # Use median for more stability
    median_interval = np.median(intervals)
    good = intervals[(intervals > 0.65 * median_interval) & (intervals < 1.35 * median_interval)]

    if len(good) < 1:
        return None

    bpm = 60.0 / np.mean(good)

    if bpm < 40 or bpm > 180:
        return None

    return round(float(bpm), 1)

class HeartSensorService:
    def __init__(self, fs=100):
        self.fs = fs
        self.sensor = MAX30102()
        self.lock = threading.Lock()
        self.ir_buf = deque(maxlen=fs * 12)
        self.latest = {"ok": False, "bpm": None, "spo2": None, "finger": False, "ir_dc": None, "samples": 0, "ts": None, "status": "no_signal"}

    def start(self):
        self.sensor.init()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                samples = self.sensor.read_fifo_samples()
                if samples:
                    with self.lock:
                        for _, ir in samples: self.ir_buf.append(ir)
                        ir_list = list(self.ir_buf)
                        bpm = compute_bpm_from_ir(ir_list, fs=self.fs)
                        ir_dc = float(np.mean(ir_list[-self.fs * 2:])) if len(ir_list) >= self.fs * 2 else float(np.mean(ir_list))
                        finger = ir_dc > 5000

                        status = "no_signal"
                        if finger:
                            status = "measuring" if bpm is None else ("low" if bpm < 50 else ("high" if bpm > 120 else "normal"))

                        self.latest.update({"ok": True, "bpm": bpm, "finger": finger, "ir_dc": round(ir_dc, 2), "samples": len(self.ir_buf), "ts": time.time(), "status": status})
                        with sensor_state_lock:
                            latest_sensor_state["heart"] = dict(self.latest)
                time.sleep(0.03)
            except Exception as e:
                with self.lock: self.latest.update({"ok": False, "error": str(e), "status": "error"})
                with sensor_state_lock: latest_sensor_state["heart"] = dict(self.latest)
                time.sleep(0.2)

    def get_status(self):
        with self.lock: return dict(self.latest)

heart_service = HeartSensorService(fs=100)

def heart_uploader():
    while True:
        try:
            data = heart_service.get_status()
            ref_heart().set({
                "bpm": data.get("bpm"), "status": data.get("status"),
                "connected": bool(data.get("finger")), "finger": bool(data.get("finger")),
                "ir_dc": data.get("ir_dc"), "samples": data.get("samples"), "timestamp": int(time.time())
            })
        except Exception: pass
        time.sleep(3)

def system_uploader():
    while True:
        try:
            with sensor_state_lock:
                payload = {
                    "camera_running": latest_sensor_state["system"]["camera_running"],
                    "camera_available": latest_sensor_state["system"]["camera_available"],
                    "firebase_ok": latest_sensor_state["system"]["firebase_ok"],
                    "heart_available": latest_sensor_state["system"]["heart_available"],
                    "started_at": latest_sensor_state["system"]["started_at"],
                    "last_seen": int(time.time())
                }
            ref_system().set(payload)
        except Exception: pass
        time.sleep(5)
