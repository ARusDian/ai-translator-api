FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --default-timeout=300 -r requirements.txt --disable-pip-version-check

# === Pre-download HuggingFace models for offline use ===
RUN mkdir -p models/opus-mt-id-en && \
    mkdir -p models/nllb && \
    python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='Helsinki-NLP/opus-mt-id-en', cache_dir='models', local_dir='models/opus-mt-id-en'); \
snapshot_download(repo_id='facebook/nllb-200-distilled-600M', cache_dir='models', local_dir='models/nllb')"


COPY . .

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "src.main:app"]
