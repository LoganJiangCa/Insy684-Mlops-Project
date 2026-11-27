"""Prediction records shared with the monitoring pipeline."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Float, Integer, String, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    request_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    features: Mapped[list[float]] = mapped_column(JSONB)
    prediction: Mapped[int] = mapped_column(Integer)
    true_label: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_name: Mapped[str] = mapped_column(String(255))
    model_version: Mapped[str] = mapped_column(String(64))
    inference_ms: Mapped[float] = mapped_column(Float)
