# Stage 1: Build React frontend
FROM node:22-slim AS frontend
WORKDIR /app
COPY platform/frontend/package.json platform/frontend/package-lock.json ./
RUN npm ci
COPY platform/frontend/ ./
RUN npm run build

# Stage 2: Python app
FROM python:3.12-slim
WORKDIR /app

# System deps for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install CUDA PyTorch (cu128 for RTX 5090)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

# Python deps
COPY platform/requirements.txt .
COPY Kronos/requirements.txt ./kronos-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r kronos-requirements.txt

# Copy application code
COPY platform/ ./platform/
COPY Kronos/ ./Kronos/

# Copy built frontend into platform
COPY --from=frontend /app/dist ./platform/frontend/dist

WORKDIR /app/platform

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
