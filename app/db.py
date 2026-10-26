"""SQLAlchemy engine / session helpers (plain SQLAlchemy 2.0, SQLite by default)."""
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


def init_db(database_url: str, create_tables: bool = True):
    """Create the engine (+ tables) and return a session factory."""
    is_sqlite = database_url.startswith("sqlite")
    engine = create_engine(database_url, future=True,
                           connect_args={"check_same_thread": False} if is_sqlite else {})
    if is_sqlite:
        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record):  # noqa: ANN001
            cur = dbapi_connection.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
    from . import models  # noqa: F401  (registers the tables on Base.metadata)
    if create_tables:
        Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory):
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
