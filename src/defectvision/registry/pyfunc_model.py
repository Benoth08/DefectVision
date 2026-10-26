"""Enveloppe MLflow 'pyfunc' : le jeu de modeles devient chargeable par mlflow.pyfunc.load_model.

Entree : DataFrame avec une colonne 'image_base64' (ou 'image_path') et,
optionnellement, une colonne 'model' (supervised, autoencoder, hybrid).
Sortie : DataFrame avec label, score, seuil et revue operateur.
"""

from __future__ import annotations

import base64

import mlflow.pyfunc
import pandas as pd


class DefectVisionModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from defectvision.inference.predictor import Predictor

        self.predictor = Predictor(context.artifacts["models"], device="cpu")

    def predict(self, context, model_input, params=None):
        from defectvision.data.preprocessing import decode_image, read_gray

        frame = pd.DataFrame(model_input)
        rows = []
        for _, row in frame.iterrows():
            if "image_base64" in row and isinstance(row["image_base64"], str):
                gray = decode_image(base64.b64decode(row["image_base64"]))
            else:
                gray = read_gray(row["image_path"])
            model = row.get("model") or (params or {}).get("model", "hybrid")
            result = self.predictor.predict(gray, model=model)
            rows.append({k: result[k] for k in ("model", "label", "score", "threshold", "review_required")})
        return pd.DataFrame(rows)
