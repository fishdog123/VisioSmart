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
    tts_playing,
    AUDIO_RETRY_SEC,
    AUDIO_MAX_RETRY_SEC,
    TTS_PRE_PLAY_DELAY_SEC,
    TTS_MIXER_BUFFER,
    TTS_START_GRACE_SEC,
    TTS_MIN_PLAY_SEC,
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
_MIXER_SETTINGS = {
    "frequency": 22050,
    "size": -16,
    "channels": 2,
    "buffer": TTS_MIXER_BUFFER,
}


def _init_mixer():
    pygame.mixer.quit()
    pygame.mixer.init(**_MIXER_SETTINGS)


def ensure_mixer():
    if pygame.mixer.get_init():
        return True

    start_time = time.time()
    delay = AUDIO_RETRY_SEC
    while True:
        try:
            _init_mixer()
            print("[TTS INFO] Pygame audio mixer initialized successfully.")
            return True
        except Exception as e:
            elapsed = time.time() - start_time
            if elapsed >= AUDIO_MAX_RETRY_SEC:
                print(f"[TTS ERROR] Mixer init failed after {elapsed:.1f}s: {e}")
                return False
            print(f"[TTS WARNING] Mixer init failed, retrying in {delay:.1f}s: {e}")
            time.sleep(delay)
            delay = min(delay * 1.5, AUDIO_MAX_RETRY_SEC)


def _play_audio_once(path):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

    start_wait = time.time()
    started = False
    while time.time() - start_wait < TTS_START_GRACE_SEC:
        if pygame.mixer.music.get_busy():
            started = True
            break
        time.sleep(0.01)

    if not started:
        return False, 0.0

    play_start = time.time()
    while pygame.mixer.music.get_busy():
        time.sleep(0.05)
        if time.time() - play_start > 20.0:  # No sentence should take > 20s
            print("[TTS Timeout] Audio driver hung up. Forcing unload.")
            break

    play_duration = time.time() - play_start
    pygame.mixer.music.unload()
    return True, play_duration


try:
    _init_mixer()
    print("[TTS INFO] Pygame audio mixer initialized successfully.")
except Exception as e:
    print(f"[TTS CRITICAL ERROR] Failed to initialize mixer: {e}")


def speak(text):
    if not voice:
        print("[TTS Error] Piper voice model not loaded.")
        return

    tts_playing.set()

    try:
        if not ensure_mixer():
            return

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

        if TTS_PRE_PLAY_DELAY_SEC > 0:
            time.sleep(TTS_PRE_PLAY_DELAY_SEC)

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

                try:
                    pygame.mixer.quit()
                except Exception:
                    pass

    except Exception as e:
        print(f"[TTS Unexpected Error] {e}")
    finally:
        tts_playing.clear()


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
