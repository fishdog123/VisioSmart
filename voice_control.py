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
    append_llm_context, get_llm_context, llm_one_shot_queue, audio_lock,
)
import llm_client

pending_llm_text = [""]
awaiting_llm_confirmation = [False]


def _handle_llm_confirmation(text):
    t = text.strip().lower()

    yes_words = {"yes", "yeah", "yep", "correct", "confirm", "right", "sure", "ok", "okay"}
    no_words = {"no", "nope", "cancel", "wrong", "stop"}

    if t in yes_words:
        user_text = pending_llm_text[0]
        pending_llm_text[0] = ""
        awaiting_llm_confirmation[0] = False

        if user_text:
            return _handle_chat_text(user_text)

        tts_queue.put("Nothing to confirm.")
        return True

    if t in no_words:
        pending_llm_text[0] = ""
        awaiting_llm_confirmation[0] = False
        tts_queue.put("Okay, say it again.")
        return True

    tts_queue.put("Please say yes or no.")
    return True

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

    if awaiting_llm_confirmation[0]:
        return _handle_llm_confirmation(text)

    words = text.split()
    # Check for commands first
    for word in set(words):
        if word in SPECIAL_COMMANDS:
            _handle_special_command(word)
            return True
        if word in VOICE_COMMANDS:
            mode_num = VOICE_COMMANDS[word]
            with mode_lock:
                if current_mode[0] is None:
                    current_mode[0] = mode_num
            if mode_num == 0:
                tts_queue.put("Exiting. Goodbye.")
            print(f"[VOICE] Recognized '{word}' -> mode {mode_num}")
            return True

    # LLM only active in Chat mode
    if active_mode_ref[0] in (GEMINI_CHAT_MODE, LOCAL_LLM_CHAT_MODE):
        pending_llm_text[0] = text
        awaiting_llm_confirmation[0] = True
        tts_queue.put("Did you say: " + text + "?")
        return True

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

    if action.get("action") == "error":
        tts_queue.put(action.get("text", "An error occurred."))
        with mode_lock:
            current_mode[0] = None
        active_mode_ref[0] = None
        print(f"[LLM] Error action: {action.get('text', '')}")
        return True

    if action.get("action") == "respond":
        response_text = action.get("text", "")
        append_llm_context("user", text)
        append_llm_context("assistant", response_text)
        tts_queue.put(response_text)
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
            print(f"[LLM] Received vision result for one-shot: {vision_result}")
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

        if final_action.get("action") == "error":
            tts_queue.put(final_action.get("text", "An error occurred."))
            with mode_lock:
                current_mode[0] = None
            active_mode_ref[0] = None
            print(f"[LLM] Error action: {final_action.get('text', '')}")
            return True

        if final_action.get("action") == "respond":
            final_speech = final_action.get("text", "")
            append_llm_context("user", text)
            append_llm_context("assistant", final_speech)
            tts_queue.put(final_speech)
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

    def listener():
        # Outer infinite loop keeps the background listener thread alive permanently
        while True:
            p = pyaudio.PyAudio()
            stream = None

            try:
                rec = vosk.KaldiRecognizer(model, 16000)
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=4000
                )
                stream.start_stream()
                print("[INFO] STT Engine connected to default system microphone node.")

            except Exception as e:
                # If mic is missing/disconnected at startup, drop here instead of crashing
                print(f"[VOICE RECUPERATION] Microphone connection failed: {e}. Retrying connection in 3 seconds...")
                p.terminate()
                time.sleep(3.0)
                continue

            # Inner stream loop reads audio frame vectors while connection is healthy
            while True:
                try:
                    with audio_lock:
                        data = stream.read(4000, exception_on_overflow=False)
                    if len(data) == 0:
                        continue

                    if rec.AcceptWaveform(data):
                        result_json = json.loads(rec.Result())
                        text = result_json.get("text", "").strip()
                        if text:
                            print(f"[VOICE] Heard (Natural End): {text}")
                            _handle_voice_text(text)

                except Exception as e:
                    # Catching the pipeline exception when the hardware unbinds from OS
                    print(f"[VOICE DISCONNECT] Microphone device context dropped or timed out: {e}")
                    break

            # Reconnection pipeline: Clean up the corrupted instance components safely
            print("[INFO] Attempting to reset mic stream channel mappings...")
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass

            p.terminate()

            # Giving WirePlumber system maps a small breathing window to register the device drop/re-add
            time.sleep(2.0)

    threading.Thread(target=listener, daemon=True).start()
