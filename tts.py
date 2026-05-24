import subprocess
import threading

from config import (
    tts_queue,
    last_spoken_text,
    TTS_VOICE,
    TTS_SPEED,
    TTS_SHUTDOWN,
    append_llm_context,
)

# ==========================================
# TTS FUNCTIONS
# ==========================================
TTS_TIMEOUT = 15  # seconds — prevents permanent hang if pw-play freezes

def speak(text):
    try:
        p1 = subprocess.Popen(["espeak-ng","-v",TTS_VOICE,"-s",str(TTS_SPEED),"--stdout",text],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p2 = subprocess.Popen(["pw-play","-"], stdin=p1.stdout, stderr=subprocess.DEVNULL)
        p1.stdout.close()
        try:
            p2.communicate(timeout=TTS_TIMEOUT)
        except subprocess.TimeoutExpired:
            p2.kill()
            p1.kill()
            p2.wait()
            print(f"[TTS Warning] Speech timed out after {TTS_TIMEOUT}s, killed subprocess")
        finally:
            p1.wait()
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
