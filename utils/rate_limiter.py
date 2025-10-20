import asyncio
import os
import time
import random
from typing import Dict, Tuple, Optional

from utils.logger import get_logger
from utils import metrics


logger = get_logger()


class _TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: int):
        self.rate = max(0.0, float(rate_per_sec))
        self.capacity = max(1, int(capacity))
        self.tokens = self.capacity
        self.updated_at = time.monotonic()
        self.lock = asyncio.Lock()

    async def take(self, amount: int = 1) -> None:
        async with self.lock:
            while True:
                now = time.monotonic()
                if self.rate > 0:
                    # refill
                    delta = now - self.updated_at
                    add = delta * self.rate
                    if add > 0:
                        self.tokens = min(self.capacity, self.tokens + add)
                        self.updated_at = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                # need to wait
                # Compute time to next token
                needed = amount - self.tokens
                wait_s = needed / self.rate if self.rate > 0 else 0.25
                wait_s = max(0.01, min(wait_s, 1.0))
                await asyncio.sleep(wait_s)


class _ConcurrencyGate:
    def __init__(self, max_concurrency: int):
        self.sem = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def acquire(self):
        await self.sem.acquire()

    def release(self):
        try:
            self.sem.release()
        except ValueError:
            pass

    def in_use(self) -> int:
        try:
            # Python 3.11 has ._value and ._waiters; we approximate in-use
            return self.sem._value  # type: ignore[attr-defined]
        except Exception:
            return 0


class RateLimiter:
    """
    Provider+model scoped limiter combining token bucket rate limiting and
    concurrency gate. Redis is not required; we provide a robust in-memory
    fallback that is process-local and conservative.

    Env vars (optional):
    - AI_RPS: max requests per second (default 1.0)
    - AI_RPM: max requests per minute (overrides capacity; default 0=disabled)
    - AI_MAX_CONCURRENCY: max concurrent requests per provider+model (default 2)
    - AI_RETRY_BASE: base backoff seconds (default 1.0)
    - AI_RETRY_CAP: cap backoff seconds (default 20.0)
    - AI_MAX_RETRIES: max retries on transient errors (default 3)
    """

    _instances: Dict[Tuple[str, str], "RateLimiter"] = {}

    def __init__(self, provider: str, model: str):
        self.provider = provider or "unknown"
        self.model = model or "unknown"

        rps = float(os.getenv("AI_RPS", "1.0"))
        rpm = int(os.getenv("AI_RPM", "0"))
        # capacity aligns with rpm if provided, else burst of 2*rps
        capacity = rpm if rpm > 0 else max(2, int(rps * 2))
        self.bucket = _TokenBucket(rate_per_sec=rps, capacity=capacity)

        max_conc = int(os.getenv("AI_MAX_CONCURRENCY", "2"))
        self.gate = _ConcurrencyGate(max_conc)

        # transient 429/5xx tracker for simple circuit breaker
        self._fail_count = 0
        self._fail_window_reset = 0.0
        self._circuit_open_until = 0.0
        self._lock = asyncio.Lock()

    @classmethod
    def get(cls, provider: str, model: str) -> "RateLimiter":
        key = (provider or "unknown", model or "unknown")
        inst = cls._instances.get(key)
        if inst is None:
            inst = cls(provider, model)
            cls._instances[key] = inst
        return inst

    def _circuit_open(self) -> bool:
        return time.monotonic() < self._circuit_open_until

    async def _trip_circuit(self, duration: float) -> None:
        async with self._lock:
            self._circuit_open_until = max(self._circuit_open_until, time.monotonic() + duration)
            try:
                metrics.record_ai_circuit_open(self.provider, self.model)
            except Exception:
                pass

    async def _record_failure(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now > self._fail_window_reset:
                # reset window each 60s
                self._fail_count = 0
                self._fail_window_reset = now + 60.0
            self._fail_count += 1
            # Open circuit if too many failures quickly
            if self._fail_count >= 5:
                await self._trip_circuit(30.0)

    async def _record_success(self) -> None:
        async with self._lock:
            self._fail_count = 0
            self._fail_window_reset = time.monotonic() + 60.0

    async def wait_for_slot(self) -> None:
        # Check simple circuit breaker
        if self._circuit_open():
            delay = max(0.0, self._circuit_open_until - time.monotonic())
            logger.warning(f"AI circuit open for {self.provider}/{self.model}, delaying {delay:.1f}s")
            await asyncio.sleep(delay)
        await self.gate.acquire()
        try:
            metrics.set_ai_request_concurrency(self.provider, self.model, -1)  # mark acquire; will set actual below
        except Exception:
            pass
        # Record active concurrency value
        try:
            metrics.set_ai_request_concurrency(self.provider, self.model, self.current_concurrency())
        except Exception:
            pass
        await self.bucket.take(1)

    def release(self) -> None:
        try:
            self.gate.release()
            metrics.set_ai_request_concurrency(self.provider, self.model, self.current_concurrency())
        except Exception:
            pass

    def current_concurrency(self) -> int:
        try:
            # Estimate current in-use = capacity - available
            # asyncio.Semaphore has _value which is available permits
            max_conc = int(os.getenv("AI_MAX_CONCURRENCY", "2"))
            available = getattr(self.gate.sem, "_value", 0)
            in_use = max(0, max_conc - int(available))
            return in_use
        except Exception:
            return 0

    @staticmethod
    def compute_backoff(attempt: int) -> float:
        base = float(os.getenv("AI_RETRY_BASE", "1.0"))
        cap = float(os.getenv("AI_RETRY_CAP", "20.0"))
        # full jitter backoff: random between 0 and min(cap, base*2^attempt)
        max_wait = min(cap, base * (2 ** max(0, attempt)))
        return random.uniform(0.0, max_wait)

    @staticmethod
    def parse_retry_after(header_val: Optional[str]) -> Optional[float]:
        if not header_val:
            return None
        try:
            # First try integer seconds
            sec = int(header_val.strip())
            return float(max(0, sec))
        except Exception:
            pass
        # Try HTTP-date
        try:
            import email.utils
            ts = email.utils.parsedate_to_datetime(header_val)
            if ts is None:
                return None
            # Convert to epoch seconds safely
            try:
                target = ts.timestamp()
            except AttributeError:
                return None
            delta = target - time.time()
            return max(0.0, float(delta))
        except Exception:
            return None

    async def on_transient_error(self, status_code: int, retry_after: Optional[str]) -> float:
        if status_code == 429:
            try:
                metrics.record_ai_rate_limit_hit(self.provider, self.model)
            except Exception:
                pass
        await self._record_failure()
        delay = self.parse_retry_after(retry_after)
        if delay is None:
            delay = self.compute_backoff(1)
        return delay

    async def on_success(self) -> None:
        await self._record_success()
