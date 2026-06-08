from collections.abc import Generator
from typing import Any

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    create_engine = None
    sessionmaker = None
    Session = Any
    SQLALCHEMY_AVAILABLE = False

    class DeclarativeBase:  # type: ignore[no-redef]
        pass

from .config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
engine = create_engine(settings.database_url, pool_pre_ping=True) if SQLALCHEMY_AVAILABLE else None
SessionLocal = (
    sessionmaker(bind=engine, autoflush=False, autocommit=False)
    if SQLALCHEMY_AVAILABLE
    else None
)


def get_db() -> Generator[Session | None, None, None]:
    if SessionLocal is None:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
