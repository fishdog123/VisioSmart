# VisioSmart - Smart Glasses For Blind People

Voice-driven computer-vision assistant for Raspberry Pi 5.
Use speech-to-text to select detection modes; the system speaks results via TTS.

Important: this README does not document the heart-sensor feature. Live video streaming is available (see "Live stream" below).

---

## Overview

This project runs on a Raspberry Pi and provides several real-time computer-vision "modes" that are selected by voice. The Pi captures frames with the Pi Camera, processes them with mode-specific detectors, and announces concise results using `pw-play`. A small Flask server provides an MJPEG live stream,snapshot endpoints and sensors data.

Key points:
- Entry Point: Main loop in `main.py`
- Voice input: offline Vosk (handled by `voice_control.py`).
- Mode selection: spoken commands map to mode numbers in `config.py`.
- Audio output: `tts.py` queues messages and plays them via `pw-play` (PipeWire).
- Live video: Flask endpoints in `main.py` serve `/video`, `/viewer`, `sensors`, and `/snapshot.jpg`.

---

## Hardware
- Raspberry Pi 5
- Pi Camera Module v3 (CSI)
- Wireless earbuds (act as both microphone and speaker; pair once in OS so they auto-connect on boot)
- 3D-printed headset mount to hold the Pi + power bank

---

## Quick start

On the Raspberry Pi:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip pipewire portaudio19-dev libsndfile1

# create a virtualenv
python3 -m venv --system-site-packages  venv
. venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python main.py --headless
```

If you prefer to run without headless mode (development with a display), omit `--headless`.

---

## Modes (what's implemented)

- Currency Detection — `modes/currency_detector.py`
- Face Recognition (with emotion detection) — `modes/face_recognizer.py`
- OCR / Text Reading — `modes/ocr_processor.py`
- Object Detection — `modes/object_detector.py`
- Chat Assistant using Gemini — `llm_client.py` + chat logic in `main.py`
- Chat Assistant using Local Qwen3.5-2B — `llm_client.py` + chat logic in `main.py`
- Scene Description using Gemini — `modes/gemini_scene.py`
- Scene Description using Local smolvlm-500m `modes/local_scene.py`
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
- "five" → Gemini Chat Assistant (mode 5)
- "six"  → Local Chat Assistant (mode 6)
- "seven" →  Gemini Scene Description (mode 7)
- "eight" →  Local Scene Description (mode 8)
- "nine" or "color" → Color Recognition (mode 9)
- "ten" or "light" → Light Recognition (mode 10)
- "stop" / "exit" / "quit" → Exit
- Special commands: `help`, `repeat`, `status` (see `voice_control.py`)

---

## Live stream

`main.py` starts a small Flask server with these endpoints (defaults shown in `config.py`):

- `/video` — MJPEG stream
- `/viewer` — simple HTML viewer that embeds `/video`
- `/snapshot.jpg` — single-frame JPEG snapshot
- `/health` — health check JSON
- `/sensors` or `metrics` - sensors data

By default the server binds `0.0.0.0:8000` (see `STREAM_HOST`/`STREAM_PORT` in `config.py`).

---

## TTS & audio

- TTS is implemented in `tts.py` using `pw-play` (PipeWire).
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

---

## Notes
- The face-embeddings pickle and some model files are expected to be produced or downloaded on the Raspberry Pi. The repo does not include large model binaries.
- Ensure the local models (Qwen and SmolVLM) are running before starting the application. The provided `ai_glasses.service` can be used to start them automatically at boot.
