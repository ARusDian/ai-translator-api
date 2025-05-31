FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt -i https://pypi.org/simple

# === Pre-download HuggingFace models for offline use ===
RUN mkdir -p models/opus-mt-id-en && \
    mkdir -p models/nllb && \
    python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='Helsinki-NLP/opus-mt-id-en', cache_dir='models', local_dir='models/opus-mt-id-en', local_dir_use_symlinks=False); \
snapshot_download(repo_id='facebook/nllb-200-distilled-600M', cache_dir='models', local_dir='models/nllb', local_dir_use_symlinks=False)"


COPY . .

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "src.main:app"]
