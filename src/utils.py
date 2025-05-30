import hashlib
import json
import logging
import os
import redis
import torch
import uuid
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bs4 import BeautifulSoup, NavigableString
from fastapi import HTTPException

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TRANSLATION_API_KEY", "supersecretapikey123")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_MAX_MEMORY_POLICY = os.getenv("REDIS_MAXMEMORY_POLICY", "allkeys-lru")

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
try:
    redis_client.config_set("maxmemory-policy", REDIS_MAX_MEMORY_POLICY)
except redis.exceptions.ResponseError:
    logging.warning("Redis maxmemory-policy could not be set.")

SUPPORTED_TARGETS = ["en", "zh", "ar", "ko"]
MODEL_MAP = {
    ("id", "en"): "models/opus-mt-id-en",
    ("en", "zh"): "models/nllb",
    ("en", "ar"): "models/nllb",
    ("en", "ko"): "models/nllb",
}
LANG_CODE_MAP = {
    "id": "ind_Latn",
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "ko": "kor_Hang",
}

model_cache = {}


def load_model(model_path):
    if model_path not in model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model_cache[model_path] = (tokenizer, model)
    return model_cache[model_path]


def make_cache_key(text: str, src: str, tgt: str) -> str:
    raw_key = f"{src}:{tgt}:{text}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def rate_limit(api_key: str, limit: int = 1000, window_seconds: int = 3600):
    key = f"ratelimit:{api_key}"
    current = redis_client.incr(key)
    if current == 1:
        redis_client.expire(key, window_seconds)
    if current > limit:
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded. Try again later."
        )


def translate_once(text: str, src: str, tgt: str) -> str:
    model_path = MODEL_MAP.get((src, tgt))
    if not model_path:
        raise ValueError(f"No model for {src} → {tgt}")
    tokenizer, model = load_model(model_path)

    if "nllb" in model_path:
        tokenizer.src_lang = LANG_CODE_MAP[src]
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(LANG_CODE_MAP[tgt])
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs, forced_bos_token_id=forced_bos_token_id
            )
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    else:
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            output = model.generate(**inputs)
        return tokenizer.decode(output[0], skip_special_tokens=True)


def translate_text(source: str, target: str, text: str) -> str:
    source = source.lower()
    target = target.lower()
    text = text.strip()

    if source == target:
        return text
    if source != "id" or target not in SUPPORTED_TARGETS:
        raise ValueError("Only id → en/zh/ar/ko supported.")

    cache_key = make_cache_key(text, source, target)
    cached = redis_client.get(cache_key)
    if cached:
        return cached.decode("utf-8")

    if (source, target) in MODEL_MAP:
        result = translate_once(text, source, target)
    elif (source, "en") in MODEL_MAP and ("en", target) in MODEL_MAP:
        intermediate = translate_once(text, source, "en")
        result = translate_once(intermediate, "en", target)
    else:
        raise ValueError("Unsupported translation path.")

    redis_client.setex(cache_key, 3600, result)
    return result


def translate_html_text(source: str, target: str, html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    def recursively_translate(node):
        for content in node.contents:
            if isinstance(content, NavigableString) and content.strip():
                translated = translate_text(source, target, content.strip())
                content.replace_with(translated)
            elif content.name is not None:
                recursively_translate(content)

    recursively_translate(soup)
    return str(soup)


def log_usage(api_key: str, endpoint: str, detail: dict):
    request_id = str(uuid.uuid4())
    date_key = datetime.utcnow().strftime("%Y-%m-%d")
    log_key = f"logs:{api_key}:{date_key}"
    log_entry = {
        "request_id": request_id,
        "time": datetime.utcnow().isoformat(),
        "endpoint": endpoint,
        "summary": detail,
    }
    redis_client.rpush(log_key, json.dumps(log_entry))
    redis_client.incr(f"count:{api_key}:{date_key}")
    logging.info(
        f"REQ_ID={request_id} | API_KEY={api_key} | endpoint={endpoint} | detail={detail}"
    )
