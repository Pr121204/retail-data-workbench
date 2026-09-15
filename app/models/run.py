import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.db import Base


def get_utc_now():
    return datetime.now(timezone.utc)


def generate_uuid():
    return str(uuid.uuid4())


class Run(Base):
    __tablename__ = "runs"

    id = Column(String, primary_key=True, default=generate_uuid)
    status = Column(String, nullable=False, default="pending")
    created_at = Column(DateTime, nullable=False, default=get_utc_now)
    updated_at = Column(DateTime, nullable=False, default=get_utc_now, onupdate=get_utc_now)
    error_message = Column(Text, nullable=True)

    datasets = relationship("Dataset", back_populates="run", cascade="all, delete-orphan")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, default=generate_uuid)
    run_id = Column(String, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    stage = Column(String, nullable=False, default="raw")
    file_path = Column(String, nullable=False)
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    profile_json = Column(Text, nullable=True)
    cleaning_plan_json = Column(Text, nullable=True)
    validation_json = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=get_utc_now)

    run = relationship("Run", back_populates="datasets")
