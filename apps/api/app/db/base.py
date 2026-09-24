from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Import model modules so Alembic sees all tables through Base.metadata.
from app.db import models as models  # noqa: E402, F401
