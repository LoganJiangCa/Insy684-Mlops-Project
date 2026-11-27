import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from register_model import register_fixture
from sqlalchemy import text

from model_api.app import Settings, create_app


@pytest.fixture(scope="session")
def database_url():
    value = os.environ.get("TEST_DATABASE_URL")
    if not value:
        pytest.fail("Set TEST_DATABASE_URL to a disposable PostgreSQL database.")
    return value


@pytest.fixture(scope="session")
def registered_model(tmp_path_factory):
    directory = tmp_path_factory.mktemp("registry")
    sample = register_fixture(directory)
    return f"sqlite:///{directory / 'mlflow.db'}", sample


@pytest.fixture
def api(database_url, registered_model):
    tracking_uri, sample = registered_model
    app = create_app(Settings(database_url, tracking_uri))
    with TestClient(app) as client:
        yield client, app, sample
        with app.state.engine.begin() as connection:
            connection.execute(text("DELETE FROM prediction_logs"))


@pytest.fixture(scope="session")
def live_sample():
    directory = Path(os.environ.get("FIXTURE_DIR", "/mlflow"))
    return json.loads((directory / "sample.json").read_text())
