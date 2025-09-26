FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV METRICS_PORT=9108
HEALTHCHECK --interval=1m --timeout=5s --retries=3 CMD ["python", "scripts/healthcheck.py"]

CMD ["python", "main.py"]
