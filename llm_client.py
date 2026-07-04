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
    LLM_INENT_TEMPERATURE,
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
# 1. System instruction only (No examples inside!)
SYSTEM_PROMPT = (
    "You are a fast smart glasses assistant for blind users.\n"
    "Rule: You must ONLY output a valid JSON object based on the user request.\n\n"
    "JSON Formats:\n"
    '1. If you can answer directly: {"action":"respond","text":"Your answer"}\n'
    '2. If you need a camera tool: {"action":"run_mode_once","mode":1|2|3|4,"reason":"Why"}\n\n'
    "Mode Map: 1=Currency, 2=Face, 3=OCR, 4=Object\n\n"
)

# 2. Structured Few-Shot Examples
ROUTER_EXAMPLES = [
    {"role": "user", "content": "who is here"},
    {"role": "assistant", "content": '{"action":"run_mode_once","mode":2,"reason":"Identify people"}'},

    {"role": "user", "content": "read this"},
    {"role": "assistant", "content": '{"action":"run_mode_once","mode":3,"reason":"Read text"}'},

    {"role": "user", "content": "how much money is this"},
    {"role": "assistant", "content": '{"action":"run_mode_once","mode":1,"reason":"Check currency"}'},

    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": '{"action":"respond","text":"Hello! How can I help you?"}'},

    {"role": "user", "content": "how much when is he"},
    {"role": "assistant", "content": '{"action":"respond","text":"I did not catch that. Please ask about a person, text, or object."}'},

    {"role": "user", "content": "what can you see"},
    {"role": "assistant", "content": '{"action":"run_mode_once","mode":4,"reason":"Describe surroundings"}'},

    {"role": "user", "content": "any money"},
    {"role": "assistant", "content": '{"action":"run_mode_once","mode":1,"reason":"Check for currency"}'},
]

FINALIZE_PROMPT = (
    "You are an intelligent smart glasses assistant for blind users.\n"
    'Rule: You must ONLY reply with this exact JSON format: {"action":"respond","text":"..."}\n\n'
    "Task: Answer the user's question naturally using the provided Vision Tool Data.\n"
    "Translate raw detection lists into a clean spatial description (e.g., 'right in front of you'). "
    "Be warm, brief, and do not invent facts. If nothing was detected, inform the user clearly."
)


def _post_local_llm(system_content, context, user_text, examples=None, temperature=0.0):
    """
    Accepts standardized signature parameters and maps them
    to an OpenAI-compatible Messages array for local endpoints.
    """
    messages = [{"role": "system", "content": system_content}]
    if examples:
        messages.extend(examples)

    for item in context or []:
        role = item.get("role")
        text = item.get("content", "")
        if text:
            if role in {"user", "assistant", "system"}:
                messages.append({"role": role, "content": text})

    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": LLM_TOP_P,
        "top_k": LLM_TOP_K,
        "response_format": {"type": "json_object"}
    }
    # print(f"[LOCAL LLM] Sending request: {messages}")

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


def _post_gemini(system_content, context, user_text, examples=None, temperature=0.0):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    contents = []

    if examples:
        for ex in examples:
            contents.append({
                "role": "user" if ex["role"] == "user" else "model",
                "parts": [{"text": ex["content"]}],
            })
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
            "temperature": temperature,
            "topP": LLM_TOP_P,
            "topK": LLM_TOP_K,
            "responseMimeType": "application/json",
        },
    }
    print(f"[GEMINI] Sending request with content: {contents}")

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
        try:
            print(f"[LLM] Failed to parse JSON, trying raw content: {raw}")
            data = json.loads(raw)
        except Exception:
            print(f"[LLM] Failed to parse JSON from raw content: {raw}")
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


def _trim_context(context, max_context=4):
    return context[-max_context:] if len(context) > max_context else context


def chat_once(user_text, context, active_mode):
    trimmed = _trim_context(context, max_context=2)
    if active_mode == 5:
        content = _post_gemini(SYSTEM_PROMPT, trimmed, user_text, examples=ROUTER_EXAMPLES, temperature=LLM_INENT_TEMPERATURE)
    elif active_mode == 6:
        content = _post_local_llm(SYSTEM_PROMPT, trimmed, user_text, examples=ROUTER_EXAMPLES, temperature=LLM_INENT_TEMPERATURE)
    else:
        content = _post_gemini(SYSTEM_PROMPT, trimmed, user_text, examples=ROUTER_EXAMPLES, temperature=LLM_INENT_TEMPERATURE)

    return _parse_action(content)


def finalize_response(user_text, context, active_mode, vision_result):

    user_payload = (
        f"Vision Tool Environmental Data: {vision_result}\n"
        f"User Original Query: {user_text}"
    )
    trimmed = _trim_context(context, max_context=4)

    if active_mode == 5:
        content = _post_gemini(FINALIZE_PROMPT, trimmed, user_payload, examples=None, temperature=LLM_TEMPERATURE)
    elif active_mode == 6:
        content = _post_local_llm(FINALIZE_PROMPT, trimmed, user_payload, examples=None, temperature=LLM_TEMPERATURE)
    else:
        content = _post_gemini(FINALIZE_PROMPT, trimmed, user_payload, examples=None, temperature=LLM_TEMPERATURE)

    return _parse_action(content)
