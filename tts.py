import subprocess
import threading
import hashlib
from datetime import datetime
from pathlib import Path

from config import (
    tts_queue,
    last_spoken_text,
    TTS_VOICE,
    TTS_SPEED,
    TTS_SHUTDOWN,
    append_llm_context,
    TTS_CAPTURE,
    TTS_CAPTURE_DIR,
)

# ==========================================
# TTS FUNCTIONS
# ==========================================
TTS_TIMEOUT = 30 # seconds — prevents permanent hang if pw-play freezes

def speak(text):
    try:
        # Generate audio bytes with espeak-ng
        p1 = subprocess.Popen(["espeak-ng", "-v", TTS_VOICE, "-s", str(TTS_SPEED), "--stdout", text],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        audio_bytes, _ = p1.communicate()

        # Optionally save the generated WAV for offline analysis
        if TTS_CAPTURE:
            try:
                capture_dir = Path(TTS_CAPTURE_DIR) if TTS_CAPTURE_DIR else (Path.cwd() / "tts_captures")
                capture_dir.mkdir(parents=True, exist_ok=True)
                short_hash = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:8]
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"tts_{timestamp}_{short_hash}.wav"
                out_path = capture_dir / filename
                with open(out_path, "wb") as f:
                    f.write(audio_bytes)
                print(f"[TTS Capture] Wrote {out_path}")
            except Exception as exc:
                print(f"[TTS Capture Error] {exc}")

        # Play the captured/generated audio via pw-play
        p2 = subprocess.Popen(["pw-play", "-"], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            p2.communicate(input=audio_bytes, timeout=TTS_TIMEOUT)
        except subprocess.TimeoutExpired:
            p2.kill()
            print(f"[TTS Warning] Playback timed out after {TTS_TIMEOUT}s, killed subprocess")
        finally:
            p2.wait()
    except Exception as e:
        print(f"[TTS Error] {e}")

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is TTS_SHUTDOWN:
            break
        last_spoken_text[0] = text
        append_llm_context("assistant", text)
        speak(text)
        tts_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()
