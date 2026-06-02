import os
import time
import wave
import json
import threading
import pygame
from piper import PiperVoice

from config import (
    tts_queue,
    last_spoken_text,
    TTS_SHUTDOWN,
    append_llm_context,
    audio_lock,
)

# --- SYSTEM CONFIGURATION ---
MODEL_PATH = "en_US-joe-medium.onnx"
CONFIG_PATH = "en_US-joe-medium.onnx.json"
OUTPUT_WAV = "/tmp/tts_output.wav"

# Load the voice model once globally during setup
if os.path.exists(MODEL_PATH):
    voice = PiperVoice.load(MODEL_PATH)
else:
    print(f"[TTS WARNING] Piper model not found at {MODEL_PATH}")
    voice = None

# --- NEW: STATIC MIXER INITIALIZATION ---
# Initialize the sound system ONCE at system start.
# 22050Hz is standard for Piper medium models, stereo (channels=2) forces
# PipeWire to cleanly mirror the mono voice stream to both your left and right earbuds.
try:
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
    print("[TTS INFO] Pygame audio mixer initialized successfully.")
except Exception as e:
    print(f"[TTS CRITICAL ERROR] Failed to initialize mixer: {e}")


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

        # 4. Play the track using the persistent audio channel
        with audio_lock:
            try:
                # If earbuds connected/disconnected right before this, load might choke.
                # Wrapping it in an inner try-except block guarantees stability.
                pygame.mixer.music.load(OUTPUT_WAV)
                pygame.mixer.music.play()

                # Block safely inside the lock while the audio plays out
                start_time = time.time()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
                    if time.time() - start_time > 30.0: # No sentence should take > 30s
                        print("[TTS Timeout] Audio driver hung up. Forcing unload.")
                        break
                print("[TTS] playback finished")
                # Unload immediately so the file asset isn't locked on disk
                pygame.mixer.music.unload()


            except Exception as mixer_err:
                print(f"[TTS] Mixer failed: {mixer_err}")

                try:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                    time.sleep(0.2)

                    print("[TTS] Mixer recovered")
                except Exception as e:
                    print(f"[TTS] Recovery failed: {e}")

    except Exception as e:
        print(f"[TTS Unexpected Error] {e}")


def tts_worker():
    while True:
        text = tts_queue.get()
        if text is TTS_SHUTDOWN:
            break

        last_spoken_text[0] = text
        append_llm_context("assistant", text)

        speak(text)
        tts_queue.task_done()


# Start background system thread
threading.Thread(target=tts_worker, daemon=True).start()
