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
    # piper1-gpl requires passing the model path; it automatically searches for the .json config
    voice = PiperVoice.load(MODEL_PATH)
else:
    print(f"[TTS WARNING] Piper model not found at {MODEL_PATH}")
    voice = None


def speak(text):
    if not voice:
        print("[TTS Error] Piper voice model not loaded.")
        return

    try:
        # 1. Clean up any leftover audio artifact
        if os.path.exists(OUTPUT_WAV):
            try:
                os.remove(OUTPUT_WAV)
            except OSError:
                pass

        # 2. Let Piper natively generate the complete WAV file structure
        with wave.open(OUTPUT_WAV, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

        # 3. Read the generated WAV file metadata to adapt to your earbud hardware
        with wave.open(OUTPUT_WAV, "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            
            # Prevent attempting to play empty files if synthesis is interrupted
            if frames == 0:
                print("[TTS Error] Piper synthesized an empty audio track.")
                return

        # 4. Play the track to the earbuds using an isolated Pygame lifecycle
        with audio_lock:
            # Initialize mixer dynamically using the actual file's sample rate.
            # Force channels=2 (Stereo) so PipeWire maps the mono stream to both ears.
            pygame.mixer.init(frequency=rate, size=-16, channels=2)
            
            pygame.mixer.music.load(OUTPUT_WAV)
            pygame.mixer.music.play()

            # Block safely inside the lock while the audio plays out
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)

            # 5. Unload and un-initialize the mixer immediately to free the PipeWire channel
            pygame.mixer.music.unload()
            pygame.mixer.quit()
        
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


# Start background system thread
threading.Thread(target=tts_worker, daemon=True).start()