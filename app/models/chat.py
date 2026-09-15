import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.db import Base
from app.models.run import generate_uuid, get_utc_now


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    run_id = Column(String, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, nullable=False, default=get_utc_now)
    # active_filters stores carried-over context as JSON text, e.g. {"region": "West"}
    active_filters_json = Column(Text, nullable=False, default="{}")

    turns = relationship(
        "ChatTurn",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatTurn.turn_index",
    )


class ChatTurn(Base):
    __tablename__ = "chat_turns"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    turn_index = Column(Integer, nullable=False)
    question = Column(Text, nullable=False)
    # "ok", "clarification_needed", "refused", "plan_rejected", "llm_fallback"
    status = Column(String, nullable=False)
    plan_json = Column(Text, nullable=True)        # the QueryPlan used, if any
    plan_source = Column(String, nullable=True)    # "mock", "llm", "mock_fallback"
    answer_text = Column(Text, nullable=True)      # short natural-language summary of the result
    evidence_json = Column(Text, nullable=True)    # the evidence object from execute_plan
    created_at = Column(DateTime, nullable=False, default=get_utc_now)

    session = relationship("ChatSession", back_populates="turns")
