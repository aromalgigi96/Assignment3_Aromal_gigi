# tests/test_api.py

"""
Unit tests for the Penguin Classification FastAPI app.

Covers:
- Root & health endpoints
- Valid prediction
- Missing field, invalid type, out-of-range, empty requests, boundary cases
- GCS download helper
- Internal prediction error handling
"""

import sys
import os
from pathlib import Path

# 1. Ensure tests import local app/ not any installed package
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# 2. Point tests at our model & metadata in app/data
os.environ["MODEL_PATH"] = str(project_root / "app" / "data" / "model.json")
os.environ["METADATA_PATH"] = str(project_root / "app" / "data" / "metadata.json")

from fastapi.testclient import TestClient
import pytest

# Import items from your app
from app.main import app, _download_from_gcs, model

client = TestClient(app)


def test_read_root() -> None:
    """Root endpoint returns welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello! Welcome to the Penguins Classification API."}


def test_health_check() -> None:
    """Health endpoint returns status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_input() -> None:
    """Valid input yields 200 and species key."""
    payload = {
        "bill_length_mm": 40.0,
        "bill_depth_mm": 18.0,
        "flipper_length_mm": 195,
        "body_mass_g": 4000,
        "year": 2008,
        "sex": "male",
        "island": "Biscoe"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "species" in response.json()
    assert isinstance(response.json()["species"], str)


def test_predict_missing_field() -> None:
    """Missing required field returns 400."""
    payload = {
        # Missing bill_length_mm
        "bill_depth_mm": 18.0,
        "flipper_length_mm": 195,
        "body_mass_g": 4000,
        "year": 2008,
        "sex": "male",
        "island": "Biscoe"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert isinstance(response.json().get("detail"), list)


def test_predict_invalid_type() -> None:
    """Invalid data type returns 400."""
    payload = {
        "bill_length_mm": "not_a_float",
        "bill_depth_mm": 18.0,
        "flipper_length_mm": 195,
        "body_mass_g": 4000,
        "year": 2008,
        "sex": "male",
        "island": "Biscoe"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400


@pytest.mark.parametrize("body_mass", [-3000, 0])
def test_predict_out_of_range_and_boundary(body_mass: int) -> None:
    """Out-of-range and boundary values are handled gracefully."""
    payload = {
        "bill_length_mm": 40.0,
        "bill_depth_mm": 18.0,
        "flipper_length_mm": 195,
        "body_mass_g": body_mass,
        "year": 2008,
        "sex": "male" if body_mass != 0 else "female",
        "island": "Biscoe" if body_mass != 0 else "Torgersen"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in (200, 400)


def test_predict_empty_request() -> None:
    """Empty JSON body returns 400."""
    response = client.post("/predict", json={})
    assert response.status_code == 400


def test_download_from_gcs(monkeypatch, tmp_path):
    """Test that our GCS helper downloads a blob to a temp file."""
    # Create a dummy source file
    source = tmp_path / "dummy.txt"
    content = "hello"
    source.write_text(content)

    # Dummy blob that writes our source content
    class DummyBlob:
        def __init__(self, path):
            self._path = path
        def download_to_filename(self, filename):
            Path(filename).write_text(self._path.read_text())

    class DummyBucket:
        def __init__(self, path):
            self._path = path
        def blob(self, blob_name):
            return DummyBlob(self._path)

    class DummyClient:
        def __init__(self, path):
            self._path = path
        def bucket(self, bucket_name):
            return DummyBucket(self._path)

    # Monkeypatch storage.Client in our app module
    import app.main as main_mod
    monkeypatch.setattr(main_mod, "storage", type("S", (), {"Client": lambda *args, **kwargs: DummyClient(source)}))

    uri = f"gs://any-bucket/{source.name}"
    downloaded = main_mod._download_from_gcs(uri)
    assert downloaded.read_text() == content


def test_predict_internal_error(monkeypatch):
    """Test that an exception in model.predict returns a 500 error."""
    # Monkeypatch model.predict to raise
    def fake_predict(df):
        raise RuntimeError("fail")
    monkeypatch.setattr(model, "predict", fake_predict)

    payload = {
        "bill_length_mm": 40.0,
        "bill_depth_mm": 18.0,
        "flipper_length_mm": 195,
        "body_mass_g": 4000,
        "year": 2008,
        "sex": "male",
        "island": "Biscoe"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 500
    assert response.json().get("detail") == "Internal prediction error"
