"""Tests for the FastAPI service that do not require the dataset on disk."""

import pandas as pd
from fastapi.testclient import TestClient

import energy_forecasting.api.app as app_module

client = TestClient(app_module.app)


def test_health_reports_no_data_instead_of_crashing(monkeypatch):
    def missing_dataset():
        raise FileNotFoundError("data/features/dataset.parquet")

    monkeypatch.setattr(app_module, "_get_dataset", missing_dataset)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "no_data"
    assert response.json()["dataset_rows"] == 0


def test_health_reports_row_count_when_data_is_available(monkeypatch):
    monkeypatch.setattr(app_module, "_get_dataset", lambda: pd.DataFrame({"x": [1, 2, 3]}))
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["dataset_rows"] == 3


def test_models_endpoint_works_without_data():
    response = client.get("/models")

    assert response.status_code == 200
    assert len(response.json()["models"]) > 0