from collections.abc import Generator
from contextlib import contextmanager

import redis

DEFAULT_REDIS_DB = 0


@contextmanager
def get_redis_db(db=None, **kwargs) -> Generator[redis.Redis]:
    """Connect to redis."""

    db = db or DEFAULT_REDIS_DB

    conn = redis.Redis(host="localhost", port=6379, db=db, decode_responses=True, **kwargs)
    conn.ping()

    yield conn

    conn.close()
