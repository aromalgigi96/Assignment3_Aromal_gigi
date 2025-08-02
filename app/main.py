# app/main.py

import json
import logging
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from google.cloud import storage
from pydantic import BaseModel
from starlette.status import HTTP_400_BAD_REQUEST
from xgboost import XGBClassifier

# ─── Load environment variables ───────────────────────────────────────────────────────
load_dotenv()  # reads .env in project root

# ─── 1. Enums for categorical validation ─────────────────────────────────────────────
class Island(str, Enum):
    """Valid penguin island options."""
    Torgersen = "Torgersen"
    Biscoe    = "Biscoe"
    Dream     = "Dream"

class Sex(str, Enum):
    """Valid penguin sex options."""
    male   = "male"
    female = "female"

# ─── 2. Pydantic request model ────────────────────────────────────────────────────────
class PenguinFeatures(BaseModel):
    """Request schema for /predict endpoint."""
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island

# ─── 3. App & logger setup ────────────────────────────────────────────────────────────
app = FastAPI()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("penguin-api")

# ─── 4. Validation exception handler ─────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return HTTP 400 when request validation fails."""
    logger.debug(f"Validation error on {request.url}: {exc}")
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={"detail": exc.errors()},
    )

# ─── 5. Download and load model from GCS ──────────────────────────────────────────────
bucket_name = os.environ["GCS_BUCKET_NAME"]
blob_name   = os.environ["GCS_BLOB_NAME"]

client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob(blob_name)

# Download model.json to a temp file
temp_dir = Path(tempfile.mkdtemp())
local_model_path = temp_dir / blob_name
blob.download_to_filename(str(local_model_path))
logger.info(f"Downloaded model from gs://{bucket_name}/{blob_name} to {local_model_path}")

# Load XGBoost model
model = XGBClassifier()
model.load_model(str(local_model_path))
logger.info("XGBoost model loaded from GCS")

# ─── 6. Load metadata from local file ─────────────────────────────────────────────────
metadata_path = Path(__file__).parent / "data" / "metadata.json"
with open(metadata_path, "r") as f:
    meta: Dict[str, List[str]] = json.load(f)

FEATURE_COLUMNS: List[str] = meta["feature_columns"]
LABEL_CLASSES: List[str]   = meta["label_classes"]
logger.info(f"Loaded metadata: {len(FEATURE_COLUMNS)} features, {len(LABEL_CLASSES)} classes")

# ─── 7. Root endpoint ────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def read_root() -> Dict[str, str]:
    """Root endpoint returning a welcome message."""
    return {"message": "Hello! Welcome to the Penguins Classification API."}

# ─── 8. Health-check endpoint ───────────────────────────────────────────────────────
@app.get("/health")
def health_check() -> Dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}

# ─── 9. Prediction endpoint ─────────────────────────────────────────────────────────
@app.post("/predict")
def predict(features: PenguinFeatures) -> Dict[str, Any]:
    """
    Predict the species of a penguin from provided features.

    Args:
        features (PenguinFeatures): Validated penguin features.

    Returns:
        dict: Predicted penguin species.
    """
    payload = features.model_dump()  # Pydantic v2
    logger.info(f"Prediction requested: {payload}")

    df = pd.DataFrame([payload])
    df = pd.get_dummies(df, columns=["sex", "island"])
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    try:
        pred = model.predict(df)[0]
        species = LABEL_CLASSES[int(pred)]
        logger.info(f"Prediction result: {species}")
        return {"species": species}
    except Exception:
        logger.error("Prediction failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal prediction error")
