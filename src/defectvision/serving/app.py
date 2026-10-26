"""API d'inspection DefectVision (FastAPI).

Points d'entree :
- POST /predict        : une image -> decision du modele choisi (hybrid par defaut) ;
- POST /predict/batch  : plusieurs images ;
- GET  /models         : jeu de modeles en service, seuils et metriques ;
- GET  /health, /ready : sondes de vie et de disponibilite ;
- GET  /metrics        : metriques Prometheus ;
- POST /admin/reload   : recharge le champion MLflow (jeton X-Admin-Token).

Lancement : uvicorn defectvision.serving.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import hmac
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from defectvision import __version__
from defectvision.data.preprocessing import decode_image
from defectvision.serving import monitoring
from defectvision.serving.model_store import ModelStore
from defectvision.serving.schemas import BatchPrediction, Health, ModelName, Prediction, Readiness, ReloadResult
from defectvision.serving.settings import Settings
from defectvision.utils import configure_torch_threads

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
settings = Settings()
store = ModelStore(settings)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_torch_threads()
    await run_in_threadpool(store.load)
    yield


app = FastAPI(
    title="DefectVision API",
    version=__version__,
    description="Controle qualite visuel de pieces moulees : modeles supervise, non supervise et hybride.",
    root_path=settings.root_path,
    lifespan=lifespan,
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        path = getattr(route, "path", "inconnue")
        if path != "/metrics":
            monitoring.HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
            monitoring.HTTP_LATENCY.labels(path).observe(time.perf_counter() - start)


def _predictor():
    if store.predictor is None:
        monitoring.PREDICTION_ERRORS.labels("modele_absent").inc()
        raise HTTPException(503, detail=f"aucun modele charge : {store.last_error}")
    return store.predictor


async def _read_image(file: UploadFile):
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        monitoring.PREDICTION_ERRORS.labels("trop_volumineux").inc()
        raise HTTPException(413, detail=f"image trop volumineuse (max {settings.max_upload_bytes} octets)")
    try:
        gray = decode_image(data)
    except ValueError as exc:
        monitoring.PREDICTION_ERRORS.labels("image_illisible").inc()
        raise HTTPException(400, detail=f"{file.filename}: {exc}") from exc
    monitoring.observe_input(gray)
    return gray


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url=f"{settings.root_path}/docs")


@app.get("/health", response_model=Health, tags=["sondes"])
def health():
    return Health(status="ok")


@app.get("/ready", response_model=Readiness, tags=["sondes"])
def ready(response: Response):
    p = store.predictor
    if p is None:
        response.status_code = 503
    return Readiness(ready=p is not None, model_version=p.version if p else None, source=store.source,
                     error=store.last_error)


@app.get("/models", tags=["modeles"])
def models():
    return {"source": store.source, **_predictor().describe()}


@app.post("/predict", response_model=Prediction, response_model_exclude_none=True, tags=["prediction"])
async def predict(
    file: UploadFile = File(..., description="Image de la piece (JPEG, PNG, BMP)"),
    model: ModelName = Query("hybrid", description="Modele a utiliser"),
    explain: bool = Query(False, description="Joindre les cartes de chaleur (Grad-CAM, anomalie)"),
):
    predictor = _predictor()
    gray = await _read_image(file)
    result = await run_in_threadpool(predictor.predict, gray, model, explain)
    monitoring.observe_prediction(result)
    return Prediction(filename=file.filename, **result)


@app.post("/predict/batch", response_model=BatchPrediction, response_model_exclude_none=True, tags=["prediction"])
async def predict_batch(
    files: list[UploadFile] = File(..., description="Images des pieces"),
    model: ModelName = Query("hybrid"),
):
    if len(files) > settings.max_batch_files:
        monitoring.PREDICTION_ERRORS.labels("lot_trop_grand").inc()
        raise HTTPException(413, detail=f"au plus {settings.max_batch_files} images par lot")
    predictor = _predictor()
    predictions = []
    for file in files:
        gray = await _read_image(file)
        result = await run_in_threadpool(predictor.predict, gray, model, False)
        monitoring.observe_prediction(result)
        predictions.append(Prediction(filename=file.filename, **result))
    return BatchPrediction(model=model, count=len(predictions), defects=sum(p.is_defect for p in predictions),
                           predictions=predictions)


@app.post("/admin/reload", response_model=ReloadResult, tags=["administration"])
async def reload_models(x_admin_token: str = Header("", alias="X-Admin-Token")):
    if not settings.admin_token or not hmac.compare_digest(x_admin_token, settings.admin_token):
        raise HTTPException(403, detail="jeton d'administration invalide")
    ok = await run_in_threadpool(store.load)
    p = store.predictor
    return ReloadResult(reloaded=ok, model_version=p.version if p else None, source=store.source,
                        error=None if ok else store.last_error)


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
