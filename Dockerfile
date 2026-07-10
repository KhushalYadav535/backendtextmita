# ─── Base Image ───────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ─── System Dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ─── Working Directory ────────────────────────────────────────────────────────
WORKDIR /app

# ─── Install Python Dependencies ─────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# ─── Copy App Code ────────────────────────────────────────────────────────────
COPY ocr_server.py .
COPY download_models.py .
COPY hf_robust_download.py .

# ─── Download PaddleOCR Models (cached via Docker volume) ────────────────────
# Models store at ~/.paddleocr — mounted as volume so NOT re-downloaded on rebuild
RUN python download_models.py || echo "Models will be downloaded on first run"

# ─── Expose Port ─────────────────────────────────────────────────────────────
EXPOSE 5000

# ─── Start Server ────────────────────────────────────────────────────────────
CMD gunicorn --workers 2 --bind 0.0.0.0:${PORT:-5000} --timeout 600 ocr_server:app
