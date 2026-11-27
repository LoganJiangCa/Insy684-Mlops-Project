"""A real HTTP call to the API container, then an independent PostgreSQL read."""

import os
from uuid import UUID

import httpx
import pytest
from sqlalchemy import create_engine, text


@pytest.mark.skipif(not os.getenv("API_URL"), reason="Run with compose.api-test.yml")
def test_running_api_container(live_sample, database_url):
    with httpx.Client(base_url=os.environ["API_URL"], timeout=30) as client:
        assert client.get("/health").status_code == 200
        response = client.post(
            "/predict",
            json={
                "features": live_sample["features"],
                "true_label": live_sample["true_label"],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] == live_sample["expected_prediction"]
        assert body["model_version"] == "1"
        assert client.post("/predict", json={"features": []}).status_code == 422
        metrics = client.get("/metrics").text
        assert 'path="/predict",status="200"' in metrics
        assert 'path="/predict",status="422"' in metrics
        assert "http_request_duration_seconds_bucket" in metrics
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT features, prediction, true_label, model_version "
                    "FROM prediction_logs WHERE request_id = :request_id"
                ),
                {"request_id": UUID(body["request_id"])},
            ).one()
            assert row.features == live_sample["features"]
            assert row.prediction == body["prediction"]
            assert row.true_label == live_sample["true_label"]
            assert row.model_version == "1"
    finally:
        engine.dispose()
