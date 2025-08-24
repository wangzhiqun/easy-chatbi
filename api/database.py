from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from utils import get_config, logger

config = get_config()

engine = create_engine(
    config.mysql_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    try:
        from . import models

        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized")

    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise
