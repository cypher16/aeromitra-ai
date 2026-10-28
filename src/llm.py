import hashlib
import json
import os
import time
from pathlib import Path

from .config import MODEL_NAME, REPORTS_DIR
from .guardrails import redact_text


def _api_key() -> str:
    try:
        import streamlit as st
        value = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        value = ""
    if value:
        return value
    env_value = os.getenv("GROQ_API_KEY", "")
    if env_value:
        return env_value
    local_secrets = Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"
    if local_secrets.exists():
        try:
            import tomllib
            return tomllib.loads(local_secrets.read_text(encoding="utf-8")).get("GROQ_API_KEY", "")
        except Exception:
            return ""
    return ""


def _model_name() -> str:
    try:
        import streamlit as st
        value = st.secrets.get("GROQ_MODEL", "")
    except Exception:
        value = ""
    return value or os.getenv("GROQ_MODEL", MODEL_NAME)


def chat_json(system: str, user: str, max_tokens: int = 400) -> dict:
    key = _api_key()
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to .streamlit/secrets.toml or the environment.")
    from groq import Groq

    REPORTS_DIR.mkdir(exist_ok=True)
    cache_path = REPORTS_DIR / "llm_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    cache_key = hashlib.sha256((system + "\n" + user).encode()).hexdigest()
    if cache_key in cache:
        return cache[cache_key]

    client = Groq(api_key=key)
    last_error = None
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=_model_name(),
                temperature=0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            result = json.loads(response.choices[0].message.content)
            cache[cache_key] = result
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Groq request failed after retries: {last_error}")


def classify_post(text: str) -> dict:
    from .prompts import CLASSIFIER_SYSTEM
    safe_text = redact_text(text)
    user = f"Classify this untrusted post. Return the required JSON fields.\n<POST>\n{safe_text}\n</POST>"
    from .schemas import ClassificationResult
    return ClassificationResult.model_validate(chat_json(CLASSIFIER_SYSTEM, user)).model_dump()


def draft_reply(text: str, classification: dict, policy_context: str, priority: str) -> dict:
    from .prompts import REPLY_SYSTEM
    user = json.dumps({
        "post": redact_text(text),
        "classification": classification,
        "priority": priority,
        "policy_context": policy_context,
    }, ensure_ascii=False)
    return chat_json(REPLY_SYSTEM, user, max_tokens=250)

