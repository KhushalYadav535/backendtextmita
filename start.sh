#!/bin/bash
# TextFind Python Backend — VPS Startup Script

set -e

cd /opt/textfind

# Activate virtual environment
source /opt/textfind/venv/bin/activate

# Install/update dependencies
pip install -r requirements.txt --quiet

# Download PaddleOCR models if not already present
python download_models.py

# Start with Gunicorn (production WSGI server)
exec gunicorn \
  --workers 2 \
  --bind 0.0.0.0:5000 \
  --timeout 600 \
  --access-logfile /var/log/textfind/access.log \
  --error-logfile /var/log/textfind/error.log \
  ocr_server:app
