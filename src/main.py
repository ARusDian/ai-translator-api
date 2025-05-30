from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import Response
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from datetime import datetime
from dotenv import load_dotenv
import json

from src.models import TranslateRequest, TranslateHtmlRequest
from src.utils import (
    API_KEY,
    SUPPORTED_TARGETS,
    rate_limit,
    translate_text,
    translate_html_text,
    log_usage,
)

load_dotenv()

app = FastAPI(title="Indonesian Translation API with Redis and HTML Support (NLLB)")

REQUEST_COUNT = Counter(
    "translation_api_requests_total", "Total API Requests", ["endpoint"]
)


def verify_api_key(request: Request) -> str:
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Missing or invalid Authorization header"
        )
    token = auth.split("Bearer ")[-1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return token


@app.post(
    "/translate",
    summary="Translate plain text from Indonesian",
    description="Translate Indonesian text into one of the supported target languages: English, Chinese, Arabic, or Korean.",
)
def api_translate(req: TranslateRequest, api_key: str = Depends(verify_api_key)):
    rate_limit(api_key)
    REQUEST_COUNT.labels(endpoint="/translate").inc()
    try:
        translated = translate_text("id", req.target_lang, req.text)
        log_usage(
            api_key, "/translate", {"lang": req.target_lang, "chars": len(req.text)}
        )
        return {"translated_text": translated}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/translate_html",
    summary="Translate HTML content from Indonesian",
    description="Translate all visible text within HTML markup from Indonesian into the specified target language.",
)
def api_translate_html(
    req: TranslateHtmlRequest, api_key: str = Depends(verify_api_key)
):
    rate_limit(api_key)
    REQUEST_COUNT.labels(endpoint="/translate_html").inc()
    try:
        translated = translate_html_text("id", req.target_lang, req.html)
        log_usage(
            api_key,
            "/translate_html",
            {"lang": req.target_lang, "chars": len(req.html)},
        )
        return {"translated_html": translated}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/logs/{api_key}",
    summary="Get translation usage logs",
    description="Fetch daily logs and request count for a given API key. Optional query parameter 'date' can be used to specify the log date (default is today, UTC). The response includes a list of logged requests and total request count for that day.",
)
def get_logs(api_key: str, date: str = None):
    from src.utils import redis_client  # local import to avoid circular dependency

    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    if not date:
        date = datetime.utcnow().strftime("%Y-%m-%d")

    log_key = f"logs:{api_key}:{date}"
    count_key = f"count:{api_key}:{date}"

    logs = redis_client.lrange(log_key, 0, -1)
    parsed_logs = [json.loads(log.decode()) for log in logs]
    count = int(redis_client.get(count_key) or 0)

    return {
        "api_key": api_key,
        "date": date,
        "total_requests": count,
        "logs": parsed_logs,
    }


@app.get("/list_supported_languages")
def list_supported_languages():
    return {
        "input_language": "id",
        "supported_target_languages": SUPPORTED_TARGETS,
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.on_event("startup")
def preload_models():
    from src.utils import MODEL_MAP, load_model

    for model_path in set(MODEL_MAP.values()):
        load_model(model_path)
