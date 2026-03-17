FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/oms/

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

CMD ["sh", "-c", "python -m uvicorn oms.main:app --host 0.0.0.0 --port ${PORT}"]
