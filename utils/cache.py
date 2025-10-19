import json
import os
import time
from typing import Any, Optional
from utils.logger import get_logger


logger = get_logger()


class Cache:
    """
    Simple cache wrapper with optional Redis backend.
    Falls back to in-memory TTL cache if Redis is unavailable.
    """

    def __init__(self, namespace: str = "scores"):
        self.namespace = namespace
        self.enabled = os.getenv("ENABLE_CACHE", "true").lower() in {"1", "true", "yes", "on"}
        self._mem: dict[str, tuple[float, str]] = {}
        self._redis = None

        if not self.enabled:
            logger.info("Cache disabled via ENABLE_CACHE env")
            return

        # Try Redis if importable and reachable
        try:
            import redis  # type: ignore

            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                client = redis.Redis.from_url(redis_url, decode_responses=True)
            else:
                host = os.getenv("REDIS_HOST", "localhost")
                port = int(os.getenv("REDIS_PORT", "6379"))
                db = int(os.getenv("REDIS_DB", "0"))
                client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

            # ping to ensure connectivity
            client.ping()
            self._redis = client
            logger.info("Redis cache enabled for namespace '%s'", self.namespace)
        except Exception as e:
            # Redis not available, use in-memory
            logger.warning("Redis unavailable, using in-memory cache: %s", str(e))
            self._redis = None

    def _full_key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None

        full_key = self._full_key(key)

        if self._redis is not None:
            try:
                data = self._redis.get(full_key)
                if data is None:
                    return None
                return json.loads(data)
            except Exception as e:
                logger.warning("Redis GET failed, falling back to memory: %s", str(e))

        # memory fallback
        now = time.time()
        item = self._mem.get(full_key)
        if not item:
            return None
        expiry, payload = item
        if now > expiry:
            # expired
            self._mem.pop(full_key, None)
            return None
        try:
            return json.loads(payload)
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        if not self.enabled:
            return

        full_key = self._full_key(key)
        payload = json.dumps(value, ensure_ascii=False)

        if self._redis is not None:
            try:
                self._redis.set(full_key, payload, ex=ttl_seconds)
                return
            except Exception as e:
                logger.warning("Redis SET failed, falling back to memory: %s", str(e))

        # memory fallback
        expiry = time.time() + ttl_seconds
        self._mem[full_key] = (expiry, payload)
