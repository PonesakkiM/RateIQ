"""
RateIQ – SQLAlchemy ORM models & DB setup
Extended with ChatLog and CompetitorAnalysisLog tables.
"""
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, Text, DateTime, String
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.core.config import settings

engine = create_engine(settings.db_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id             = Column(Integer, primary_key=True, index=True)
    input_features = Column(Text, nullable=False)   # JSON string
    prediction     = Column(Float, nullable=False)
    confidence     = Column(Float, nullable=False)
    trend_adjusted = Column(Float, nullable=True)   # trend-adjusted rating
    analysis_json  = Column(Text, nullable=True)    # full analysis result JSON
    timestamp      = Column(DateTime, default=datetime.utcnow)

    def set_features(self, features: dict):
        self.input_features = json.dumps(features)

    def get_features(self) -> dict:
        return json.loads(self.input_features)


class ChatLog(Base):
    """Stores each chat turn for history and analytics."""
    __tablename__ = "chat_logs"

    id              = Column(Integer, primary_key=True, index=True)
    query           = Column(Text, nullable=False)
    response        = Column(Text, nullable=False)          # full response JSON
    detected_intents = Column(Text, nullable=True)          # JSON array of strings
    app_context     = Column(Text, nullable=True)           # JSON: app data snapshot
    timestamp       = Column(DateTime, default=datetime.utcnow)


class CompetitorAnalysisLog(Base):
    """Stores competitor analysis results."""
    __tablename__ = "competitor_analysis_logs"

    id              = Column(Integer, primary_key=True, index=True)
    app_data        = Column(Text, nullable=False)           # JSON input
    analysis_result = Column(Text, nullable=False)           # JSON result
    category        = Column(String(100), nullable=True)
    timestamp       = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
