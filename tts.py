import os
import wave
import threading
import subprocess
from piper import PiperVoice
from config import  MODEL_PATH, OUTPUT_WAV

from config import (
    tts_queue,
    last_spoken_text,
    TTS_SHUTDOWN,
    append_llm_context,
    audio_lock,
)

# Load the voice model once globally during setup
if os.path.exists(MODEL_PATH):
    print(f"[TTS INFO] Loading Piper voice model from {MODEL_PATH}...")
    voice = PiperVoice.load(MODEL_PATH)
    print("[TTS INFO] Piper model loaded successfully.")
else:
    print(f"[TTS CRITICAL ERROR] Piper model not found at {MODEL_PATH}")
    voice = None


def speak(text):
    if not voice:
        print("[TTS Error] Piper voice model not loaded.")
        return

    try:
        # 1. Clean up any leftover audio artifact safely
        if os.path.exists(OUTPUT_WAV):
            try:
                os.remove(OUTPUT_WAV)
            except OSError:
                pass

        # 2. Let Piper natively generate the complete WAV file structure
        with wave.open(OUTPUT_WAV, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

        # 3. Quick verification check on the file asset
        if not os.path.exists(OUTPUT_WAV) or os.path.getsize(OUTPUT_WAV) < 44:
            print("[TTS Error] Piper generated an empty or corrupt audio track.")
            return

        # 4. Play the track natively using the OS audio server
        with audio_lock:
            try:
                # If Bluetooth reconnects, PipeWire instantly routes it correctly.
                subprocess.run(["pw-play", OUTPUT_WAV], check=True)

            except FileNotFoundError:
                # Safe fallback to standard ALSA player if pw-play isn't found
                subprocess.run(["aplay", "-q", OUTPUT_WAV], check=True)

            except subprocess.CalledProcessError as e:
                print(f"[TTS Playback Error] Command failed: {e}")

            except Exception as playback_err:
                print(f"[TTS Playback Error] Unexpected failure: {playback_err}")

    except Exception as e:
        print(f"[TTS Unexpected Error] {e}")


def tts_worker():
    while True:
        text = tts_queue.get()
        if text is TTS_SHUTDOWN:
            break

        last_spoken_text[0] = text
        # append_llm_context("assistant", text)

        speak(text)
        tts_queue.task_done()


# Start background system thread
threading.Thread(target=tts_worker, daemon=True).start()
