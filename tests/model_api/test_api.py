from unittest.mock import patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from model_api.app import PredictionRequest, Settings, create_app
from model_api.storage import PredictionLog


def test_registered_model_prediction_is_persisted(api):
    client, app, sample = api
    response = client.post(
        "/predict",
        json={
            "features": sample["features"],
            "true_label": sample["true_label"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == sample["expected_prediction"]
    assert body["model_name"] == "digits-decision-tree"
    assert body["model_version"] == "1"
    with Session(app.state.engine) as session:
        row = session.get(PredictionLog, UUID(body["request_id"]))
        assert row.features == sample["features"]
        assert row.prediction == body["prediction"]
        assert row.true_label == sample["true_label"]
        assert row.created_at.tzinfo is not None
        assert row.inference_ms >= 0


@pytest.mark.parametrize("length", [0, 63, 65])
def test_wrong_feature_count_has_no_database_side_effect(api, length):
    client, app, _ = api
    assert client.post("/predict", json={"features": [0] * length}).status_code == 422
    with Session(app.state.engine) as session:
        assert session.scalar(select(PredictionLog)) is None


@pytest.mark.parametrize("value", [True, "2", None, float("nan"), float("inf")])
def test_features_must_be_finite_numbers(value):
    with pytest.raises(ValueError):
        PredictionRequest(features=[value] + [0] * 63)


def test_drift_inputs_are_accepted_and_label_can_be_unknown(api):
    client, app, _ = api
    response = client.post("/predict", json={"features": [20.0] * 64})
    assert response.status_code == 200
    with Session(app.state.engine) as session:
        row = session.get(PredictionLog, UUID(response.json()["request_id"]))
        assert row.features == [20.0] * 64
        assert row.true_label is None


@pytest.mark.parametrize("label", [-1, 10, True, "1"])
def test_invalid_true_labels_are_rejected(api, label):
    client, _, _ = api
    assert (
        client.post(
            "/predict",
            json={
                "features": [0] * 64,
                "true_label": label,
            },
        ).status_code
        == 422
    )


def test_database_failure_does_not_return_success(api):
    client, _, _ = api
    with patch(
        "model_api.app.Session", side_effect=OperationalError("", {}, Exception())
    ):
        response = client.post("/predict", json={"features": [0] * 64})
    assert response.status_code == 503
    assert "could not be saved" in response.json()["detail"]
    assert 'status="503"' in client.get("/metrics").text


def test_inference_failure_returns_error_and_does_not_log(api):
    client, app, _ = api
    with patch.object(
        app.state.model, "predict", side_effect=ValueError("private details")
    ):
        response = client.post("/predict", json={"features": [0] * 64})
    assert response.status_code == 500
    assert "private details" not in response.text
    with Session(app.state.engine) as session:
        assert session.scalar(select(PredictionLog)) is None


def test_metrics_include_requests_latency_and_validation_errors(api):
    client, _, _ = api
    client.post("/predict", json={"features": [0] * 64})
    client.post("/predict", json={"features": []})
    client.get("/unrecognised/path")
    metrics = client.get("/metrics").text
    assert (
        'http_requests_total{method="POST",path="/predict",status="200"} 1.0' in metrics
    )
    assert (
        'http_errors_total{method="POST",path="/predict",status="422"} 1.0' in metrics
    )
    assert (
        'http_request_duration_seconds_count{method="POST",path="/predict"} 2.0'
        in metrics
    )
    assert 'path="__unmatched__"' in metrics
    assert 'path="/unrecognised/path"' not in metrics
    assert 'path="/metrics"' not in metrics


def test_health_checks_database_and_model(api):
    client, app, _ = api
    assert client.get("/health").json()["model_name"] == "digits-decision-tree"
    with patch.object(
        app.state.engine, "connect", side_effect=OperationalError("", {}, Exception())
    ):
        assert client.get("/health").status_code == 503


def test_missing_registered_version_fails_startup(database_url, registered_model):
    tracking_uri, _ = registered_model
    with (
        pytest.raises(MlflowException, match="not found"),
        TestClient(
            create_app(Settings(database_url, tracking_uri, model_version="999"))
        ),
    ):
        pass
