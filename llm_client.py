import json
import requests

from config import (
    GEMINI_API_URL,
    GEMINI_MODEL,
    GEMINI_API_KEY,
    LLM_URL,
    LLM_MODEL,
    LLM_TIMEOUT_SEC,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_TOP_K,
    LLM_MAX_CONTEXT_CHARS,
    HEADLESS_MODE,
    MODE_NAMES,
)

# ==========================================
# LLM Client (Gemini & Local VLM/LLM)
# ==========================================
SYSTEM_PROMPT = (
    "You are a fast smart glasses assistant for blind users.\n"
    "Rule: You must ONLY output a valid JSON object based on the user request.\n\n"
    "JSON Formats:\n"
    '1. If you can answer directly: {"action":"respond","text":"Your answer"}\n'
    '2. If you need a camera tool: {"action":"run_mode_once","mode":1|2|3|4,"reason":"Why"}\n\n'
    "Mode Map: 1=Currency, 2=Face, 3=OCR, 4=Object\n\n"
    "Examples:\n"
    'User: who is here\n'
    'Assistant: {"action":"run_mode_once","mode":2,"reason":"Identify people"}\n'
    'User: read this\n'
    'Assistant: {"action":"run_mode_once","mode":3,"reason":"Read text"}\n'
    'User: how much money is this\n'
    'Assistant: {"action":"run_mode_once","mode":1,"reason":"Check currency"}\n'
    'User: hello\n'
    'Assistant: {"action":"respond","text":"Hello! How can I help you?"}\n'
    'User: how much when is he\n'
    'Assistant: {"action":"respond","text":"I did not catch that. Please ask about a person, text, or object."}\n\n'
    "Respond in exactly 1 sentence. No conversational filler."
)

FINALIZE_PROMPT = (
    "You are an intelligent smart glasses assistant for blind users.\n"
    'Rule: You must ONLY reply with this exact JSON format: {"action":"respond","text":"..."}\n\n'
    "Task: Answer the user's question naturally using the provided Vision Tool Data.\n"
    "Translate raw detection lists into a clean spatial description (e.g., 'right in front of you'). "
    "Be warm, brief, and do not invent facts. If nothing was detected, inform the user clearly."
)


def _post_local_llm(system_content, context, user_text):
    """
    Accepts standardized signature parameters and maps them
    to an OpenAI-compatible Messages array for local endpoints.
    """
    messages = [{"role": "system", "content": system_content}]

    for item in context or []:
        role = item.get("role")
        text = item.get("content", "")
        if text:
            # Match standard API key roles (user / assistant)
            messages.append({
                "role": "user" if role == "user" else "assistant",
                "content": text
            })

    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "top_k": LLM_TOP_K,
    }

    try:
        response = requests.post(LLM_URL, json=payload, timeout=LLM_TIMEOUT_SEC)
        response.raise_for_status()
    except Exception as e:
        print(f"[LOCAL LLM] Error: {e}")
        return "__ERROR__"
    data = response.json()
    content = data["choices"][0]["message"]["content"].strip()

    if not HEADLESS_MODE:
        print(f"[LOCAL LLM] Raw: {content}")
    return content


def _post_gemini(system_content, context, user_text):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    contents = []
    for item in context or []:
        role = item.get("role")
        text = item.get("content", "")
        if not text:
            continue
        contents.append({
            "role": "user" if role == "user" else "model",
            "parts": [{"text": text}],
        })

    contents.append({
        "role": "user",
        "parts": [{"text": user_text}],
    })

    payload = {
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": system_content}],
        },
        "contents": contents,
        "generationConfig": {
            "temperature": LLM_TEMPERATURE,
            "topP": LLM_TOP_P,
            "topK": LLM_TOP_K,
            "responseMimeType": "application/json",
        },
    }

    url = f"{GEMINI_API_URL}/{GEMINI_MODEL}:generateContent"
    try:
        response = requests.post(
            url,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=LLM_TIMEOUT_SEC,
        )
        response.raise_for_status()
    except Exception as e:
        print(f"[GEMINI] Error: {e}")
        return "__ERROR__"
    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        return ""
    content = "".join(part.get("text", "") for part in parts).strip()
    if not HEADLESS_MODE:
        print(f"[GEMINI] Raw: {content}")
    return content


def _extract_json(content):
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content.replace("json", "", 1).strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return content[start:end + 1]


def _parse_action(content):
    if content == "__ERROR__":
        return {"action": "error", "text": "Sorry, there was an error processing your request."}
    if not content:
        return {"action": "respond", "text": "Sorry, I did not get a response."}
    raw = content
    json_str = _extract_json(content)
    if json_str:
        raw = json_str
    try:
        data = json.loads(raw)
    except Exception:
        return {"action": "respond", "text": "Sorry, I did not understand that."}

    action = data.get("action")
    if action == "run_mode_once":
        mode = data.get("mode")
        try:
            if isinstance(mode, int) and mode in MODE_NAMES:
                return {"action": "run_mode_once", "mode": mode, "reason": data.get("reason", "")}
        except Exception:
            pass
    if action == "respond":
        text = data.get("text", "").strip()
        if text:
            return {"action": "respond", "text": text}

    return {"action": "respond", "text": "Sorry, I did not understand that."}


def _trim_context(context, system_content, user_text):
    if not context:
        return []

    base_len = len(system_content) + len(user_text)
    if base_len >= LLM_MAX_CONTEXT_CHARS:
        return []

    budget = LLM_MAX_CONTEXT_CHARS - base_len
    trimmed = []
    running = 0
    for item in reversed(context):
        content = item.get("content", "")
        if not content:
            continue
        item_len = len(content) + 10
        if running + item_len > budget:
            break
        trimmed.append(item)
        running += item_len

    return list(reversed(trimmed))


def chat_once(user_text, context, active_mode):
    system_parts = [SYSTEM_PROMPT]
    if active_mode:
        system_parts.append(f"Current active mode: {MODE_NAMES.get(active_mode, 'None')}.")
    system_content = " ".join(system_parts)
    trimmed = _trim_context(context, system_content, user_text)

    if active_mode == 5:
        content = _post_gemini(system_content, trimmed, user_text)
    elif active_mode == 6:
        # content = _post_local_llm(system_content, trimmed, user_text)
        content = _post_local_llm(system_content, [], user_text)

    else:
        content = _post_gemini(system_content, trimmed, user_text)

    return _parse_action(content)


def finalize_response(user_text, context, active_mode, vision_result):
    system_parts = [FINALIZE_PROMPT]
    if active_mode:
        system_parts.append(f"Current active mode: {MODE_NAMES.get(active_mode, 'None')}.")
    system_parts.append(f"Vision result: {vision_result}")
    system_content = " ".join(system_parts)
    trimmed = _trim_context(context, system_content, user_text)

    if active_mode == 5:
        content = _post_gemini(system_content, trimmed, user_text)
    elif active_mode == 6:
        # content = _post_local_llm(system_content, trimmed, user_text)
        content = _post_local_llm(system_content, [], user_text)
    else:
        content = _post_gemini(system_content, trimmed, user_text)

    return _parse_action(content)
