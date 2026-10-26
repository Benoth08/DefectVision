# Image "trainer" : execute le pipeline DVC (dvc repro) et l'enregistrement MLflow.
# Lancee a la demande par Airflow (DockerOperator), le depot etant monte en /workspace.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace/src \
    TORCH_HOME=/workspace/.cache/torch \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

# Certificats racine supplementaires (facultatif) : si un proxy ou un antivirus
# inspecte le HTTPS du poste, deposer son certificat public (.crt) dans
# docker/certs/. Sans fichier .crt, cette etape ne change rien.
COPY docker/certs/ /usr/local/share/ca-certificates/extra/

# git : DVC s'appuie sur le depot git monte.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

COPY requirements/core.txt requirements/train.txt requirements/torch-cpu.txt /tmp/requirements/
RUN pip install -r /tmp/requirements/torch-cpu.txt \
    && pip install -r /tmp/requirements/train.txt

WORKDIR /workspace
CMD ["dvc", "repro"]
