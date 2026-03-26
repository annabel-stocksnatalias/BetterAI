import pytest
import redis


@pytest.fixture(autouse=True, name="db")
def redis_fixture(monkeypatch):
    """Change default redis connection db."""

    db = redis.StrictRedis(host="localhost", port=6379, db=1, decode_responses=True)
    monkeypatch.setattr("database.redis.redis.DEFAULT_REDIS_DB", 1)

    db.flushdb()
    yield db
    db.flushdb()
    db.close()
