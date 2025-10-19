from __future__ import annotations

import time
from typing import Optional

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
    from prometheus_client import CONTENT_TYPE_LATEST
    from prometheus_client import generate_latest, multiprocess
except Exception:  # pragma: no cover - prometheus optional
    Counter = Histogram = Gauge = None  # type: ignore
    CollectorRegistry = None  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    def generate_latest(registry=None):  # type: ignore
        return b""
    multiprocess = None  # type: ignore

from fastapi import FastAPI, Request, Response
from starlette.responses import Response as StarletteResponse
from starlette.routing import Match


_registry: Optional[CollectorRegistry] = None


# Metrics objects (filled if prometheus_client is available)
if CollectorRegistry is not None:
    _registry = CollectorRegistry()
    if multiprocess is not None:
        try:
            multiprocess.MultiProcessCollector(_registry)
        except Exception:
            pass

    API_LATENCY = Histogram(
        "api_latency_seconds",
        "API request latency in seconds",
        ["path", "method", "status"],
        registry=_registry,
        buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 5)
    )
    CACHE_HITS = Counter(
        "cache_hits_total", "Cache hits", ["namespace", "backend"], registry=_registry
    )
    CACHE_MISSES = Counter(
        "cache_misses_total", "Cache misses", ["namespace", "backend"], registry=_registry
    )
    BACKTEST_DURATION = Histogram(
        "backtest_job_duration_seconds",
        "Duration of backtest jobs",
        ["status"],
        registry=_registry,
        buckets=(0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60)
    )
    MEMORY_RSS = Gauge(
        "process_memory_rss_bytes", "Process resident memory size in bytes", registry=_registry
    )
else:
    API_LATENCY = None
    CACHE_HITS = None
    CACHE_MISSES = None
    BACKTEST_DURATION = None
    MEMORY_RSS = None


def instrument_fastapi(app: FastAPI) -> None:
    if API_LATENCY is None:
        return

    @app.middleware("http")
    async def _metrics_middleware(request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status_code = getattr(response, "status_code", 500)
        except Exception:
            status_code = 500
            raise
        finally:
            elapsed = time.perf_counter() - start
            # Resolve matching route path, else raw path
            route_path = request.url.path
            try:
                for route in app.router.routes:
                    match, _ = route.matches(request.scope)
                    if match == Match.FULL:
                        route_path = getattr(route, "path", route_path)
                        break
            except Exception:
                pass
            if API_LATENCY is not None:
                API_LATENCY.labels(path=route_path, method=request.method, status=str(status_code)).observe(elapsed)
        return response

    # Simple memory gauge update on startup and each request if available
    if MEMORY_RSS is not None:
        try:
            import os
            import psutil  # type: ignore
            process = psutil.Process(os.getpid())
            MEMORY_RSS.set(process.memory_info().rss)

            @app.middleware("http")
            async def _mem_middleware(request: Request, call_next):
                resp = await call_next(request)
                try:
                    MEMORY_RSS.set(process.memory_info().rss)
                except Exception:
                    pass
                return resp
        except Exception:
            pass


def mount_metrics_endpoint(app: FastAPI, path: str = "/metrics") -> None:
    if CollectorRegistry is None:
        return

    @app.get(path)
    def metrics() -> Response:
        return StarletteResponse(generate_latest(_registry), media_type=CONTENT_TYPE_LATEST)


def record_cache_hit(namespace: str, backend: str) -> None:
    if CACHE_HITS is not None:
        try:
            CACHE_HITS.labels(namespace=namespace, backend=backend).inc()
        except Exception:
            pass


def record_cache_miss(namespace: str, backend: str) -> None:
    if CACHE_MISSES is not None:
        try:
            CACHE_MISSES.labels(namespace=namespace, backend=backend).inc()
        except Exception:
            pass


def observe_backtest_duration(seconds: float, status: str = "ok") -> None:
    if BACKTEST_DURATION is not None:
        try:
            BACKTEST_DURATION.labels(status=status).observe(seconds)
        except Exception:
            pass
