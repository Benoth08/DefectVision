# Image "api" : service d'inference FastAPI (CPU), utilisateur non root.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    MODEL_DIR=/models \
    MODEL_CACHE_DIR=/home/app/.cache/defectvision

# Certificats racine supplementaires (facultatif), voir docker/trainer/Dockerfile.
COPY docker/certs/ /usr/local/share/ca-certificates/extra/
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/core.txt requirements/serve.txt requirements/torch-cpu.txt /tmp/requirements/
RUN pip install -r /tmp/requirements/torch-cpu.txt \
    && pip install -r /tmp/requirements/serve.txt

RUN useradd --create-home --uid 10001 app
COPY src/ /app/src/
USER app
WORKDIR /app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "defectvision.serving.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
