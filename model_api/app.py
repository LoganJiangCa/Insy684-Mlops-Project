import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated
from uuid import uuid4

import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from model_api.storage import Base, PredictionLog


@dataclass(frozen=True)
class Settings:
    database_url: str
    tracking_uri: str
    model_name: str = "digits-decision-tree"
    model_version: str = "1"
    cors_origins: tuple[str, ...] = ()


Pixel = Annotated[StrictFloat | StrictInt, Field(allow_inf_nan=False)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: list[Pixel] = Field(min_length=64, max_length=64)
    true_label: Annotated[StrictInt, Field(ge=0, le=9)] | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings(
            database_url=os.environ["DATABASE_URL"],
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
            model_name=os.getenv("MODEL_NAME", "digits-decision-tree"),
            model_version=os.getenv("MODEL_VERSION", "1"),
            cors_origins=tuple(
                origin.strip()
                for origin in os.getenv("CORS_ORIGINS", "").split(",")
                if origin.strip()
            ),
        )
    database_url = make_url(settings.database_url)
    if database_url.get_backend_name() != "postgresql":
        raise ValueError("DATABASE_URL must use PostgreSQL.")
    database_url = database_url.set(drivername="postgresql+psycopg")
    if not settings.model_version.isdecimal():
        raise ValueError("MODEL_VERSION must be a numeric registered version.")

    @asynccontextmanager
    async def lifespan(app):
        engine = create_engine(
            database_url, pool_pre_ping=True, connect_args={"connect_timeout": 5}
        )
        try:
            mlflow.set_tracking_uri(settings.tracking_uri)
            model_uri = f"models:/{settings.model_name}/{settings.model_version}"
            model = mlflow.sklearn.load_model(model_uri)
            if getattr(model, "n_features_in_", None) != 64:
                raise ValueError("The registered model must accept 64 digits features.")
            Base.metadata.create_all(engine)
            app.state.engine = engine
            app.state.model = model
            yield
        finally:
            engine.dispose()

    app = FastAPI(title="Digits prediction API", lifespan=lifespan)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
    registry = CollectorRegistry()
    requests = Counter(
        "http_requests_total",
        "HTTP requests",
        ["method", "path", "status"],
        registry=registry,
    )
    errors = Counter(
        "http_errors_total",
        "HTTP responses with status >= 400",
        ["method", "path", "status"],
        registry=registry,
    )
    duration = Histogram(
        "http_request_duration_seconds",
        "End-to-end HTTP request duration",
        ["method", "path"],
        registry=registry,
    )

    @app.middleware("http")
    async def record_metrics(request: Request, call_next):
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            path = route.path if route else "__unmatched__"
            if path != "/metrics":
                labels = (request.method, path, str(status))
                requests.labels(*labels).inc()
                duration.labels(request.method, path).observe(
                    time.perf_counter() - started
                )
                if status >= 400:
                    errors.labels(*labels).inc()

    @app.get("/health")
    def health():
        try:
            with app.state.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except SQLAlchemyError:
            raise HTTPException(503, "Prediction database unavailable.") from None
        return {
            "status": "ok",
            "model_name": settings.model_name,
            "model_version": settings.model_version,
        }

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(
            generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST}
        )

    @app.post("/predict")
    def predict(payload: PredictionRequest):
        started = time.perf_counter()
        try:
            output = np.asarray(app.state.model.predict([payload.features]))
            if output.shape != (1,) or output[0] not in range(10):
                raise ValueError("Invalid digits prediction.")
            prediction = int(output[0])
        except (ValueError, TypeError, RuntimeError):
            raise HTTPException(500, "Model inference failed.") from None
        inference_ms = (time.perf_counter() - started) * 1000
        request_id = uuid4()
        try:
            with Session(app.state.engine) as session, session.begin():
                session.add(
                    PredictionLog(
                        request_id=request_id,
                        features=[float(value) for value in payload.features],
                        prediction=prediction,
                        true_label=payload.true_label,
                        model_name=settings.model_name,
                        model_version=settings.model_version,
                        inference_ms=inference_ms,
                    )
                )
        except SQLAlchemyError:
            raise HTTPException(
                503, "Prediction could not be saved. Please retry."
            ) from None
        return {
            "request_id": str(request_id),
            "prediction": prediction,
            "model_name": settings.model_name,
            "model_version": settings.model_version,
        }

    return app
