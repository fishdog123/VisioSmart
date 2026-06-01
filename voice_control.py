import json
import os
import time
import threading
import queue
import pyaudio
import speech_recognition as sr
import vosk

from config import (
    current_mode, mode_lock, tts_queue,
    VOICE_COMMANDS, SPECIAL_COMMANDS, MODE_NAMES, VOSK_MODEL_PATH,
    active_mode_ref, last_spoken_text, LOCAL_LLM_CHAT_MODE, GEMINI_CHAT_MODE,
    append_llm_context, get_llm_context, llm_one_shot_queue,
)
import llm_client

# ==========================================
# VOICE CONTROL
# ==========================================
def _handle_special_command(word):
    """Handle non-mode voice commands: help, repeat, status/mode."""
    if word == "help":
        tts_queue.put(
            "Available commands: say one or currency for currency detection, "
            "two or face for face recognition, three or text for OCR, "
            "four or object for object detection, five or chat for assistant, "
            "six or scene for scene description, or say color (or colour) for color recognition. "
            "Say stop to exit, repeat to hear last message, status to hear current mode."
        )
    elif word == "repeat":
        last = last_spoken_text[0]
        if last:
            tts_queue.put(last)
        else:
            tts_queue.put("Nothing to repeat.")
    elif word in ("status", "mode"):
        mode = active_mode_ref[0]
        if mode and mode in MODE_NAMES:
            tts_queue.put(f"Current mode is {MODE_NAMES[mode]}")
        else:
            tts_queue.put("No mode is active.")
    print(f"[VOICE] Special command: '{word}'")

def _handle_voice_text(text):
    text = text.strip().lower()
    words = text.split()
    
    # Check for commands first
    for word in words:
        if word in SPECIAL_COMMANDS:
            _handle_special_command(word)
            return True
        if (active_mode_ref[0] == GEMINI_CHAT_MODE or active_mode_ref[0] == LOCAL_LLM_CHAT_MODE) and word in ("chat", "assistant", "five"):
            continue
        if word in VOICE_COMMANDS:
            # Don't trigger a mode change if we are trying to talk to the chat assistant
            if (active_mode_ref[0] == GEMINI_CHAT_MODE or active_mode_ref[0] == LOCAL_LLM_CHAT_MODE) and word in ("chat", "assistant", "five"):
                continue
                
            mode_num = VOICE_COMMANDS[word]
            with mode_lock:
                current_mode[0] = mode_num
            if mode_num == 0:
                tts_queue.put("Exiting. Goodbye.")
            print(f"[VOICE] Recognized '{word}' -> mode {mode_num}")
            return True

    # LLM only active in Chat mode
    if (active_mode_ref[0] == GEMINI_CHAT_MODE or active_mode_ref[0] == LOCAL_LLM_CHAT_MODE):
        return _handle_chat_text(text)

    return False


def _handle_chat_text(text):
    if not text:
        return False

    context = get_llm_context()
    active_mode = active_mode_ref[0]

    try:
        action = llm_client.chat_once(text, context, active_mode)
    except Exception as e:
        print(f"[LLM] Error: {e}")
        tts_queue.put("LLM not available.")
        return True

    if action.get("action") == "respond":
        append_llm_context("user", text)
        tts_queue.put(action.get("text", ""))
        return True

    if action.get("action") == "run_mode_once":
        response_queue = queue.Queue(maxsize=1)
        request = {"mode": action.get("mode"), "response_queue": response_queue}
        try:
            llm_one_shot_queue.put_nowait(request)
        except queue.Full:
            tts_queue.put("System is busy. Please try again.")
            return True

        try:
            vision_result = response_queue.get(timeout=3)
        except queue.Empty:
            tts_queue.put("Sorry, I could not get a result from the camera.")
            return True

        try:
            context = get_llm_context()
            final_action = llm_client.finalize_response(
                text,
                context,
                active_mode,
                vision_result,
            )
        except Exception as e:
            print(f"[LLM] Finalize error: {e}")
            tts_queue.put("LLM finalized response failed.")
            return True

        if final_action.get("action") == "respond":
            # Context updated only after a successful complete lifecycle
            append_llm_context("user", text) 
            tts_queue.put(final_action.get("text", ""))
        else:
            tts_queue.put("I could not answer that.")
        return True

    tts_queue.put("I could not answer that.")
    return True
def start_voice_listener():
    vosk.SetLogLevel(-1)

    if not os.path.exists(VOSK_MODEL_PATH):
        print(f"[WARNING] Vosk model not found at {VOSK_MODEL_PATH}")
        tts_queue.put("Warning: Voice model not found. Voice control is disabled.")
        return

    model = vosk.Model(VOSK_MODEL_PATH)
    # 16000Hz is the native rate for most lightweight Vosk models
    rec = vosk.KaldiRecognizer(model, 16000)

    def listener():
        p = pyaudio.PyAudio()
        try:
            # Open standard microphone input stream
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4000 # ~0.25 seconds of audio per read
            )
            stream.start_stream()
        except Exception as e:
            print(f"[ERROR] Failed to open microphone stream: {e}")
            tts_queue.put("Error: Microphone could not be started.")
            p.terminate()
            return

        print("[INFO] Voice control active. Listening continuously and naturally...")

        while True:
            try:
                # Read raw audio data from the mic buffer
                # exception_on_overflow=False prevents crashes on slow machines
                data = stream.read(4000, exception_on_overflow=False)
                
                if len(data) == 0:
                    continue

                # Feed data chunks instantly to Vosk
                if rec.AcceptWaveform(data):
                    # Vosk detected a natural pause/end of a phrase!
                    result_json = json.loads(rec.Result())
                    text = result_json.get("text", "").strip()
                    
                    if text:
                        print(f"[VOICE] Heard (Natural End): {text}")
                        _handle_voice_text(text)
                else:
                    # Optional: You can get interim/partial text here if you want 
                    # to show live captions, but DO NOT call FinalResult() here.
                    pass

            except Exception as e:
                print(f"[VOICE] Stream error: {e}")
                break

        # Cleanup if the loop breaks
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        p.terminate()

    threading.Thread(target=listener, daemon=True).start()