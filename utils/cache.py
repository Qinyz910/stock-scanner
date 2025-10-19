import json
import os
import time
import functools
import asyncio
from typing import Any, Optional, Callable
from utils.logger import get_logger
from utils import metrics


logger = get_logger()


class Cache:
    """
    Simple cache wrapper with optional Redis backend.
    Falls back to in-memory TTL cache if Redis is unavailable.

    Extra capabilities:
    - cache hit/miss Prometheus metrics
    - cacheable decorator for functions (sync/async)
    - health probe
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

    # ---- internals ----
    def _full_key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def _backend_name(self) -> str:
        if not self.enabled:
            return "disabled"
        return "redis" if self._redis is not None else "memory"

    # ---- basic ops ----
    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None

        full_key = self._full_key(key)

        if self._redis is not None:
            try:
                data = self._redis.get(full_key)
                if data is None:
                    metrics.record_cache_miss(self.namespace, "redis")
                    return None
                metrics.record_cache_hit(self.namespace, "redis")
                return json.loads(data)
            except Exception as e:
                logger.warning("Redis GET failed, falling back to memory: %s", str(e))

        # memory fallback
        now = time.time()
        item = self._mem.get(full_key)
        if not item:
            metrics.record_cache_miss(self.namespace, "memory")
            return None
        expiry, payload = item
        if now > expiry:
            # expired
            self._mem.pop(full_key, None)
            metrics.record_cache_miss(self.namespace, "memory")
            return None
        try:
            metrics.record_cache_hit(self.namespace, "memory")
            return json.loads(payload)
        except Exception:
            metrics.record_cache_miss(self.namespace, "memory")
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

    # ---- decorator ----
    def cacheable(self, ttl_seconds: int = 300, key: Optional[str] = None,
                  key_builder: Optional[Callable[..., str]] = None):
        """
        Decorator to cache function results. Supports sync and async functions.
        Key defaults to function name + args json.
        """
        def decorator(func):
            is_coro = asyncio.iscoroutinefunction(func)

            def build_key(args, kwargs) -> str:
                if key:
                    return key
                if key_builder:
                    return key_builder(*args, **kwargs)
                try:
                    return f"{func.__module__}.{func.__name__}:" + json.dumps(
                        {"args": args, "kwargs": kwargs}, sort_keys=True, default=str, ensure_ascii=False
                    )
                except Exception:
                    return f"{func.__module__}.{func.__name__}:{hash(str(args)+str(kwargs))}"

            if not is_coro:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    k = build_key(args, kwargs)
                    cached = self.get(k)
                    if cached is not None:
                        return cached
                    result = func(*args, **kwargs)
                    try:
                        self.set(k, result, ttl_seconds)
                    except Exception:
                        pass
                    return result
                return sync_wrapper

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                k = build_key(args, kwargs)
                cached = self.get(k)
                if cached is not None:
                    return cached
                result = await func(*args, **kwargs)
                try:
                    self.set(k, result, ttl_seconds)
                except Exception:
                    pass
                return result

            return async_wrapper

        return decorator

    # ---- health ----
    def health(self) -> dict:
        if not self.enabled:
            return {"enabled": False, "backend": "disabled", "ok": True}
        if self._redis is not None:
            try:
                self._redis.ping()
                return {"enabled": True, "backend": "redis", "ok": True}
            except Exception as e:
                return {"enabled": True, "backend": "redis", "ok": False, "error": str(e)}
        return {"enabled": True, "backend": "memory", "ok": True}
