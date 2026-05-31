import json
import os
import time
import threading
import queue

import speech_recognition as sr
import vosk

from config import (
    current_mode, mode_lock, tts_queue,
    VOICE_COMMANDS, SPECIAL_COMMANDS, MODE_NAMES, VOSK_MODEL_PATH,
    active_mode_ref, last_spoken_text, CHAT_MODE,
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
    for word in text.split():
        if word in SPECIAL_COMMANDS:
            _handle_special_command(word)
            return True
        if active_mode_ref[0] == CHAT_MODE and word in ("chat", "assistant", "five"):
            continue
        if word in VOICE_COMMANDS:
            mode_num = VOICE_COMMANDS[word]
            with mode_lock:
                current_mode[0] = mode_num
            if mode_num == 0:
                tts_queue.put("Exiting. Goodbye.")
            print(f"[VOICE] Recognized '{word}' -> mode {mode_num}")
            return True

    # LLM only active in Chat mode
    if active_mode_ref[0] == CHAT_MODE:
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
        append_llm_context("user", text)
        response_queue = queue.Queue(maxsize=1)
        request = {"mode": action.get("mode"), "response_queue": response_queue}
        try:
            llm_one_shot_queue.put_nowait(request)
        except queue.Full:
            tts_queue.put("Busy. Try again.")
            return True

        try:
            vision_result = response_queue.get(timeout=3)
        except queue.Empty:
            tts_queue.put("Sorry, I could not get a result.")
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
            tts_queue.put("LLM not available.")
            return True

        if final_action.get("action") == "respond":
            tts_queue.put(final_action.get("text", ""))
        else:
            tts_queue.put("I could not answer that.")
        return True

    tts_queue.put("I could not answer that.")
    return True

def start_voice_listener():
    vosk.SetLogLevel(-1)

    # Check if Vosk model exists
    if not os.path.exists(VOSK_MODEL_PATH):
        print(f"[WARNING] Vosk model not found at {VOSK_MODEL_PATH}")
        print("[INFO] Voice control disabled")
        tts_queue.put("Warning: Voice model not found. Voice control is disabled.")
        return

    vosk_model = vosk.Model(VOSK_MODEL_PATH)

    def listener():
        try:
            recognizer = sr.Recognizer()
            mic = sr.Microphone()
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception as e:
            print(f"[ERROR] Microphone initialization failed: {e}")
            tts_queue.put("Error: Microphone not found. Voice control is disabled.")
            return

        print("[INFO] Voice control active. Say 'one', 'two', 'three', 'four', 'five', 'six', 'seven' or 'stop'.")

        kaldi_rec = vosk.KaldiRecognizer(vosk_model, 16000)

        while True:
            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
                kaldi_rec.AcceptWaveform(raw)
                result = json.loads(kaldi_rec.FinalResult())
                text = result.get("text", "").strip()
                if text:
                    print(f"[VOICE] Heard: {text}")
                    _handle_voice_text(text)
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"[VOICE] Error: {e}")
                tts_queue.put("Microphone error. Retrying.")
                try:
                    mic = sr.Microphone()
                    with mic as source:
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                except Exception:
                    pass
                time.sleep(2)

    threading.Thread(target=listener, daemon=True).start()
