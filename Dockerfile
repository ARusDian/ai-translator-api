FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt -i https://pypi.org/simple

# === Pre-download HuggingFace models for offline use ===
RUN mkdir -p models/opus-mt-id-en && \
    mkdir -p models/nllb-200-distilled-600M && \
    python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
               AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-id-en', cache_dir='models/opus-mt-id-en'); \
               AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-id-en', cache_dir='models/opus-mt-id-en'); \
               AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M', cache_dir='models/nllb'); \
               AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M', cache_dir='models/nllb')"

COPY . .

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "src.main:app"]
