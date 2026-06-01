# Smart Glasses For Blind People

Voice-driven computer-vision assistant for Raspberry Pi 5 (Trixie).
Use offline speech-to-text to select detection modes; the system speaks results via TTS.

Important: this README does not document the heart-sensor feature. Live video streaming is available (see "Live stream" below).

---

## Overview

This project runs on a Raspberry Pi and provides several real-time computer-vision "modes" that are selected by voice. The Pi captures frames with the Pi Camera, processes them with mode-specific detectors, and announces concise results using `espeak-ng` + PipeWire. A small Flask server provides an MJPEG live stream and snapshot endpoints.

Key points:
- Voice input: offline Vosk (handled by `voice_control.py`).
- Mode selection: spoken commands map to mode numbers in `config.py`.
- Audio output: `tts.py` queues messages and plays them via `espeak-ng` piped to `pw-play` (PipeWire).
- Live video: Flask endpoints in `main.py` serve `/video`, `/viewer`, and `/snapshot.jpg`.

---

## Hardware (tested / expected)
- Raspberry Pi 5 (Trixie OS)
- Pi Camera Module v3 (CSI)
- Wireless earbuds (act as both microphone and speaker; pair once in OS so they auto-connect on boot)
- 3D-printed headset mount to hold the Pi + power bank

---

## Quick start (minimal)

On the Raspberry Pi (recommended):

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip espeak-ng pipewire portaudio19-dev libsndfile1

# create a virtualenv
python3 -m venv venv
. venv/bin/activate

# Install dependencies after generating a pinned requirements.txt on the Pi (see Requirements below)
pip install -r requirements.txt

# Test the app (headless)
python main.py --headless
```

If you prefer to run without headless mode (development with a display), omit `--headless`.

---

## Modes (what's implemented)

- Currency Detection — `modes/currency_detector.py`
- Face Recognition (with optional emotion detection) — `modes/face_recognizer.py`
- OCR / Text Reading — `modes/ocr_processor.py`
- Object Detection — `modes/object_detector.py`
- Chat Assistant (Gemini integration) — `llm_client.py` + chat logic in `main.py`
- Scene Description (Gemini vision) — `modes/gemini_scene.py`
- Color Recognition — `modes/color_recognition.py`
- Light Recognition — `modes/light_recognition.py`

Use voice commands (below) to switch between modes; each mode announces short, useful text results.

---

## Voice commands

Commands are defined in `config.py` (`VOICE_COMMANDS`). Examples:

- "one" or "currency" → Currency Detection (mode 1)
- "two" or "face" → Face Recognition (mode 2)
- "three", "ocr", or "text" → OCR/Text Reading (mode 3)
- "four" or "object" → Object Detection (mode 4)
- "five" or "chat" → Chat Assistant (mode 5)
- "six" or "scene" → Scene Description (mode 6)
- "seven" or "color" → Color Recognition (mode 7)
- "eight" or "light" → Light Recognition (mode 8)
- "stop" / "exit" / "quit" → Exit
- Special commands: `help`, `repeat`, `status` (see `voice_control.py`)

---

## Live stream

`main.py` starts a small Flask server with these endpoints (defaults shown in `config.py`):

- `/video` — MJPEG stream
- `/viewer` — simple HTML viewer that embeds `/video`
- `/snapshot.jpg` — single-frame JPEG snapshot
- `/health` — health check JSON

By default the server binds `0.0.0.0:8081` (see `STREAM_HOST`/`STREAM_PORT` in `config.py`).

---

## TTS & audio

- TTS is implemented in `tts.py` using `espeak-ng` with audio played by `pw-play` (PipeWire).
- The project expects the earbuds or other audio device to be paired and auto-connected by the OS on boot. The earbuds act as both microphone and speaker.

System packages required for audio are installed via `apt` (see Quick start).

---

## Running as a service

The repo contains `cv_glasses.service` (a systemd unit). Typical installation steps:

```bash
sudo cp cv_glasses.service /etc/systemd/system/cv_glasses.service
# Edit ExecStart and WorkingDirectory in the unit to match your install path
sudo systemctl daemon-reload
sudo systemctl enable cv_glasses
sudo systemctl start cv_glasses
sudo journalctl -u cv_glasses -f
```

Make sure `GEMINI_API_KEY` (if using chat or scene modes) is provided via environment or `.env`.

---

## Configuration

See `config.py` for all runtime configuration and voice mappings. You can set `GEMINI_API_KEY` in an `.env` file at the project root or export it in the environment before starting the service.

Example `.env` variable (provided in `.env.example`):

```text
GEMINI_API_KEY=your_api_key_here
```

---

## Requirements & pinning (exact versions — recommended workflow)

This repository includes a minimal `requirements.in` listing the Python packages the code imports. Because some packages (notably `torch`, `ultralytics`, and other native wheels) have platform-specific builds, the recommended workflow is to generate an exact, pinned `requirements.txt` on the Raspberry Pi itself.

1. Create a virtualenv on the Pi and install the names from `requirements.in`:

```bash
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.in
```

2. Run the provided pinning script to produce `requirements.txt` with exact versions installed on the Pi:

```bash
python scripts/pin_requirements_rpi.py
```

3. Install from the generated, pinned file on any fresh environment:

```bash
pip install -r requirements.txt
```

This approach ensures the exact wheels used on the Pi are recorded (important for `torch`/`ultralytics` compatibility).

---

## Testing & troubleshooting

- Test Vosk STT (microphone): `python test_vosk.py` (see `test_vosk.py`).
- Run headless main: `python main.py --headless`.
- Inspect logs for the systemd service: `sudo journalctl -u cv_glasses -f`.
- Common issues:
  - "Vosk model not found": download the Vosk small model and place it at `VOSK_MODEL_PATH` (see `config.py`).
  - Audio issues: ensure PipeWire is running and earbuds are paired and selected as the default audio device.

---

## Files of interest
- `main.py` — entry point and server
- `config.py` — central configuration (voice mappings, paths)
- `camera.py` — Picamera2 wrapper
- `voice_control.py` — Vosk STT and command parsing
- `tts.py` — TTS queue and playback
- `llm_client.py` — Gemini client for Chat/Scene modes
- `cv_glasses.service` — systemd unit

---

## Notes
- The face-embeddings pickle and some model files are expected to be produced or downloaded on the Raspberry Pi; the repo does not include large model binaries. You mentioned you already have a prebuilt face DB and scripts on the Pi — this README references `FACE_DB_PATH` from `config.py` and does not attempt to recreate that DB here.

If you want, I can now generate a `requirements.txt` from this environment (I do not recommend it for Pi-specific native wheels) or prepare an automated model-download script. Which do you prefer?
