import json
import requests

from config import (
    GEMINI_API_URL,
    GEMINI_MODEL,
    GEMINI_API_KEY,
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
# LLM Client (Gemini)
# ==========================================
SYSTEM_PROMPT = (
    "You are a fast embedded assistant for smart glasses for blind users. "
    "Return ONLY a single JSON object and nothing else. "
    "If you can answer using available context, reply: "
    "{\"action\":\"respond\",\"text\":\"...\"}. "
    "If you need fresh visual info, reply: "
    "{\"action\":\"run_mode_once\",\"mode\":1|2|3|4,\"reason\":\"...\"}. "
    "Mode map: 1=Currency, 2=Face, 3=OCR, 4=Object. "
    "Vision results may include per-person emotions in parentheses (e.g., \"Alice (happy)\"). "
    "Use that emotional context when relevant. "
    "Do not guess; if unsure, ask a short clarification in the respond text. "
    "Keep replies short, practical, and fast (about 1 sentence)."
)

FINALIZE_PROMPT = (
    "You are a fast embedded assistant for smart glasses for blind users. "
    "Return ONLY a single JSON object and nothing else: "
    "{\"action\":\"respond\",\"text\":\"...\"}. "
    "Vision results may include per-person emotions in parentheses (e.g., \"Alice (happy)\"). "
    "Use the provided vision result to answer the user. "
    "Do not add extra facts beyond the vision result and context. "
    "If nothing is detected, say so clearly and concisely."
)


def _post(system_content, context, user_text):
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
    response = requests.post(
        url,
        params={"key": GEMINI_API_KEY},
        json=payload,
        timeout=LLM_TIMEOUT_SEC,
    )
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        return ""
    content = "".join(part.get("text", "") for part in parts).strip()
    if not HEADLESS_MODE:
        print(f"[LLM] Raw: {content}")
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
        if mode in (1, 2, 3, 4):
            return {"action": "run_mode_once", "mode": mode, "reason": data.get("reason", "")}
    if action == "respond":
        text = data.get("text", "").strip()
        if text:
            return {"action": "respond", "text": text}

    # Fallback if schema is wrong
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
    # Keep most recent context entries
    for item in reversed(context):
        content = item.get("content", "")
        if not content:
            continue
        item_len = len(content) + 10  # rough overhead for role/formatting
        if running + item_len > budget:
            break
        trimmed.append(item)
        running += item_len

    return list(reversed(trimmed))


def chat_once(user_text, context, active_mode=None):
    system_parts = [SYSTEM_PROMPT]
    if active_mode:
        system_parts.append(f"Current active mode: {MODE_NAMES.get(active_mode, 'None')}.")
    system_content = " ".join(system_parts)
    trimmed = _trim_context(context, system_content, user_text)
    content = _post(system_content, trimmed, user_text)
    return _parse_action(content)


def finalize_response(user_text, context, active_mode, vision_result):
    system_parts = [FINALIZE_PROMPT]
    if active_mode:
        system_parts.append(f"Current active mode: {MODE_NAMES.get(active_mode, 'None')}.")
    system_parts.append(f"Vision result: {vision_result}")
    system_content = " ".join(system_parts)
    trimmed = _trim_context(context, system_content, user_text)
    content = _post(system_content, trimmed, user_text)
    return _parse_action(content)
