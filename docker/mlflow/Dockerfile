# Image "mlflow" : serveur de suivi et registre de modeles.
# Metadonnees dans PostgreSQL, artefacts sur un volume servi par le serveur lui-meme.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

COPY docker/certs/ /usr/local/share/ca-certificates/extra/
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY docker/mlflow/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN useradd --create-home --uid 10002 mlflow && mkdir -p /mlflow/artifacts && chown -R mlflow /mlflow
USER mlflow
EXPOSE 5000
CMD ["sh", "-c", "exec mlflow server --backend-store-uri \"$MLFLOW_BACKEND_STORE_URI\" --artifacts-destination /mlflow/artifacts --serve-artifacts --host 0.0.0.0 --port 5000 --workers 1"]
