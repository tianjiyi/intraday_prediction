FROM python:3.12-slim

WORKDIR /app

# System deps for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install CPU PyTorch first (smaller, no CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Python deps
COPY platform/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY platform/ ./platform/
COPY Kronos/ ./Kronos/

WORKDIR /app/platform

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
