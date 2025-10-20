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

    # AI streaming metrics
    AI_STREAM_ZERO_CHUNKS = Counter(
        "ai_stream_zero_chunks_count",
        "Count of AI stream sessions that yielded zero content chunks",
        ["model", "reason"],
        registry=_registry,
    )
    AI_STREAM_FALLBACK = Counter(
        "ai_stream_fallback_count",
        "Count of AI stream sessions that triggered fallback",
        ["model", "reason"],
        registry=_registry,
    )
    AI_STREAM_DURATION = Histogram(
        "ai_stream_duration_seconds",
        "Duration of AI streaming from first request to completion",
        ["model", "outcome"],
        registry=_registry,
        buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60)
    )
    # Additional stream observability
    AI_STREAM_FRAGMENTS = Counter(
        "ai_stream_fragments_total",
        "Total number of content fragments received from upstream AI streams",
        ["provider", "model"],
        registry=_registry,
    )
    AI_STREAM_EMPTY = Counter(
        "ai_stream_empty_total",
        "Total number of upstream streams that produced zero fragments",
        ["provider", "model"],
        registry=_registry,
    )

    # Upstream connection timing/idle
    AI_UPSTREAM_TTFB = Histogram(
        "ai_upstream_ttfb_seconds",
        "Time to first byte from upstream AI provider",
        ["provider", "model"],
        registry=_registry,
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30)
    )
    AI_UPSTREAM_IDLE_TIMEOUTS = Counter(
        "ai_upstream_idle_timeouts_total",
        "Total number of upstream idle read timeouts during streaming",
        ["provider", "model"],
        registry=_registry,
    )

    # Fallbacks
    AI_FALLBACK_NON_STREAM = Counter(
        "ai_fallback_non_stream_total",
        "Total number of times non-stream fallback was used",
        ["provider", "model", "reason"],
        registry=_registry,
    )
    AI_FALLBACK_SUCCESS = Counter(
        "ai_fallback_success_total",
        "Total number of successful non-stream fallbacks that returned usable content",
        ["provider", "model", "reason"],
        registry=_registry,
    )
    AI_FALLBACK_FAIL = Counter(
        "ai_fallback_fail_total",
        "Total number of non-stream fallback attempts that still failed to return content",
        ["provider", "model", "reason"],
        registry=_registry,
    )
    AI_FALLBACK_BYTES = Counter(
        "ai_fallback_bytes_total",
        "Total number of bytes returned via non-stream fallback",
        ["provider", "model"],
        registry=_registry,
    )
    AI_STREAM_ZERO_THEN_FALLBACK = Counter(
        "ai_stream_zero_then_fallback_total",
        "Total number of times a zero-fragment stream degraded to non-stream fallback",
        ["provider", "model"],
        registry=_registry,
    )

    # New metrics for output completeness/observability
    AI_OUTPUT_CHARS = Counter(
        "ai_output_chars_total",
        "Total number of characters produced by AI outputs",
        ["provider", "model"],
        registry=_registry,
    )
    AI_MISSING_SECTIONS = Counter(
        "ai_missing_sections_total",
        "Total count of missing required sections detected",
        ["provider", "model", "section"],
        registry=_registry,
    )
    AI_AUTOCONTINUE_CALLS = Counter(
        "ai_autocontinue_calls_total",
        "Total number of auto-continue completion calls made",
        ["provider", "model"],
        registry=_registry,
    )
    AI_TRUNCATED_RESPONSES = Counter(
        "ai_truncated_responses_total",
        "Total number of truncated/short AI responses detected",
        ["provider", "model", "reason"],
        registry=_registry,
    )

    # Provider switch metrics
    AI_PROVIDER_SWITCH = Counter(
        "ai_provider_switch_total",
        "Total number of times AI provider switch was triggered during streaming",
        ["from_provider", "from_model", "to_provider", "to_model", "reason"],
        registry=_registry,
    )
    AI_PROVIDER_SWITCH_SUCCESS = Counter(
        "ai_provider_switch_success_total",
        "Total number of successful provider switches (stream continued)",
        ["to_provider", "to_model"],
        registry=_registry,
    )
    AI_PROVIDER_SWITCH_FAILED = Counter(
        "ai_provider_switch_failed_total",
        "Total number of failed provider switches (still no stream)",
        ["to_provider", "to_model", "reason"],
        registry=_registry,
    )
else:
    API_LATENCY = None
    CACHE_HITS = None
    CACHE_MISSES = None
    BACKTEST_DURATION = None
    MEMORY_RSS = None
    AI_STREAM_ZERO_CHUNKS = None
    AI_STREAM_FALLBACK = None
    AI_STREAM_DURATION = None
    AI_STREAM_FRAGMENTS = None
    AI_STREAM_EMPTY = None
    AI_UPSTREAM_TTFB = None
    AI_UPSTREAM_IDLE_TIMEOUTS = None
    AI_FALLBACK_NON_STREAM = None
    AI_FALLBACK_SUCCESS = None
    AI_FALLBACK_FAIL = None
    AI_FALLBACK_BYTES = None
    AI_STREAM_ZERO_THEN_FALLBACK = None
    AI_OUTPUT_CHARS = None
    AI_MISSING_SECTIONS = None
    AI_AUTOCONTINUE_CALLS = None
    AI_TRUNCATED_RESPONSES = None
    AI_PROVIDER_SWITCH = None
    AI_PROVIDER_SWITCH_SUCCESS = None
    AI_PROVIDER_SWITCH_FAILED = None


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


# AI stream metrics helpers (no-op if prometheus_client missing)
def record_ai_stream_zero_chunks(model: str, reason: str = "unknown") -> None:
    if AI_STREAM_ZERO_CHUNKS is not None:
        try:
            AI_STREAM_ZERO_CHUNKS.labels(model=model or "unknown", reason=reason).inc()
        except Exception:
            pass


def record_ai_stream_fallback(model: str, reason: str = "unknown") -> None:
    if AI_STREAM_FALLBACK is not None:
        try:
            AI_STREAM_FALLBACK.labels(model=model or "unknown", reason=reason).inc()
        except Exception:
            pass


def observe_ai_stream_duration(seconds: float, model: str = "unknown", outcome: str = "ok") -> None:
    if AI_STREAM_DURATION is not None:
        try:
            AI_STREAM_DURATION.labels(model=model or "unknown", outcome=outcome).observe(max(0.0, float(seconds)))
        except Exception:
            pass


# ---- Additional AI rate limit/ retry/ concurrency metrics ----
if CollectorRegistry is not None:
    try:
        AI_RATE_LIMIT_HITS = Counter(
            "ai_rate_limit_hits_total",
            "Total number of AI 429 rate limit hits",
            ["provider", "model"],
            registry=_registry,
        )
        AI_RETRIES = Counter(
            "ai_retries_total",
            "Total number of AI request retries",
            ["provider", "model", "reason"],
            registry=_registry,
        )
        AI_CIRCUIT_OPEN = Counter(
            "ai_circuit_open_total",
            "Total number of times AI circuit breaker opened",
            ["provider", "model"],
            registry=_registry,
        )
        AI_REQUEST_CONCURRENCY = Gauge(
            "ai_request_concurrency",
            "Current AI request concurrency per provider/model",
            ["provider", "model"],
            registry=_registry,
        )
        AI_FALLBACK_USED = Counter(
            "ai_fallback_used_total",
            "Total number of times AI fallback was used",
            ["provider", "model", "reason"],
            registry=_registry,
        )
        AI_ZERO_CHUNKS = Counter(
            "ai_zero_chunks_total",
            "Total number of AI stream zero-chunk occurrences",
            ["provider", "model", "reason"],
            registry=_registry,
        )
    except Exception:
        AI_RATE_LIMIT_HITS = None
        AI_RETRIES = None
        AI_CIRCUIT_OPEN = None
        AI_REQUEST_CONCURRENCY = None
        AI_FALLBACK_USED = None
        AI_ZERO_CHUNKS = None
else:
    AI_RATE_LIMIT_HITS = None
    AI_RETRIES = None
    AI_CIRCUIT_OPEN = None
    AI_REQUEST_CONCURRENCY = None
    AI_FALLBACK_USED = None
    AI_ZERO_CHUNKS = None


def record_ai_rate_limit_hit(provider: str, model: str) -> None:
    if AI_RATE_LIMIT_HITS is not None:
        try:
            AI_RATE_LIMIT_HITS.labels(provider=provider or "unknown", model=model or "unknown").inc()
        except Exception:
            pass


def record_ai_retry(provider: str, model: str, reason: str = "unknown") -> None:
    if AI_RETRIES is not None:
        try:
            AI_RETRIES.labels(provider=provider or "unknown", model=model or "unknown", reason=reason).inc()
        except Exception:
            pass


def record_ai_circuit_open(provider: str, model: str) -> None:
    if AI_CIRCUIT_OPEN is not None:
        try:
            AI_CIRCUIT_OPEN.labels(provider=provider or "unknown", model=model or "unknown").inc()
        except Exception:
            pass


def set_ai_request_concurrency(provider: str, model: str, value: int) -> None:
    if AI_REQUEST_CONCURRENCY is not None:
        try:
            if value == -1:
                # getter path will set actual value
                return
            AI_REQUEST_CONCURRENCY.labels(provider=provider or "unknown", model=model or "unknown").set(max(0, int(value)))
        except Exception:
            pass


def record_ai_fallback(provider: str, model: str, reason: str = "unknown") -> None:
    if AI_FALLBACK_USED is not None:
        try:
            AI_FALLBACK_USED.labels(provider=provider or "unknown", model=model or "unknown", reason=reason).inc()
        except Exception:
            pass


def record_ai_zero_chunks(provider: str, model: str, reason: str = "unknown") -> None:
    if AI_ZERO_CHUNKS is not None:
        try:
            AI_ZERO_CHUNKS.labels(provider=provider or "unknown", model=model or "unknown", reason=reason).inc()
        except Exception:
            pass


# ---- Output completeness metrics helpers ----

def record_ai_output_chars(provider: str, model: str, chars: int) -> None:
    if AI_OUTPUT_CHARS is not None:
        try:
            AI_OUTPUT_CHARS.labels(provider=provider or "unknown", model=model or "unknown").inc(max(0, int(chars)))
        except Exception:
            pass


def record_ai_missing_section(provider: str, model: str, section: str) -> None:
    if AI_MISSING_SECTIONS is not None:
        try:
            AI_MISSING_SECTIONS.labels(provider=provider or "unknown", model=model or "unknown", section=section or "unknown").inc()
        except Exception:
            pass


def record_ai_autocontinue_call(provider: str, model: str) -> None:
    if AI_AUTOCONTINUE_CALLS is not None:
        try:
            AI_AUTOCONTINUE_CALLS.labels(provider=provider or "unknown", model=model or "unknown").inc()
        except Exception:
            pass


def record_ai_truncated_response(provider: str, model: str, reason: str = "unknown") -> None:
    if AI_TRUNCATED_RESPONSES is not None:
        try:
            AI_TRUNCATED_RESPONSES.labels(provider=provider or "unknown", model=model or "unknown", reason=reason or "unknown").inc()
        except Exception:
            pass


def observe_ai_upstream_ttfb(provider: str, model: str, seconds: float) -> None:
    if AI_UPSTREAM_TTFB is not None:
        try:
            AI_UPSTREAM_TTFB.labels(provider=provider or "unknown", model=model or "unknown").observe(max(0.0, float(seconds)))
        except Exception:
            pass


def record_ai_upstream_idle_timeout(provider: str, model: str) -> None:
    if AI_UPSTREAM_IDLE_TIMEOUTS is not None:
        try:
            AI_UPSTREAM_IDLE_TIMEOUTS.labels(provider=provider or "unknown", model=model or "unknown").inc()
        except Exception:
            pass


def record_ai_fallback_non_stream(provider: str, model: str, reason: str = "unknown") -> None:
    if AI_FALLBACK_NON_STREAM is not None:
        try:
            AI_FALLBACK_NON_STREAM.labels(provider=provider or "unknown", model=model or "unknown", reason=reason or "unknown").inc()
        except Exception:
            pass


def record_ai_fallback_bytes(provider: str, model: str, byte_count: int) -> None:
    if AI_FALLBACK_BYTES is not None:
        try:
            AI_FALLBACK_BYTES.labels(provider=provider or "unknown", model=model or "unknown").inc(max(0, int(byte_count)))
        except Exception:
            pass


def record_ai_fallback_success(provider: str, model: str, reason: str = "unknown") -> None:
    if AI_FALLBACK_SUCCESS is not None:
        try:
            AI_FALLBACK_SUCCESS.labels(provider=provider or "unknown", model=model or "unknown", reason=reason or "unknown").inc()
        except Exception:
            pass


def record_ai_fallback_fail(provider: str, model: str, reason: str = "unknown") -> None:
    if AI_FALLBACK_FAIL is not None:
        try:
            AI_FALLBACK_FAIL.labels(provider=provider or "unknown", model=model or "unknown", reason=reason or "unknown").inc()
        except Exception:
            pass


def record_ai_stream_zero_then_fallback(provider: str, model: str) -> None:
    if AI_STREAM_ZERO_THEN_FALLBACK is not None:
        try:
            AI_STREAM_ZERO_THEN_FALLBACK.labels(provider=provider or "unknown", model=model or "unknown").inc()
        except Exception:
            pass


def record_ai_stream_fragments(provider: str, model: str, count: int) -> None:
    if AI_STREAM_FRAGMENTS is not None:
        try:
            AI_STREAM_FRAGMENTS.labels(provider=provider or "unknown", model=model or "unknown").inc(max(0, int(count)))
        except Exception:
            pass


def record_ai_stream_empty(provider: str, model: str) -> None:
    if AI_STREAM_EMPTY is not None:
        try:
            AI_STREAM_EMPTY.labels(provider=provider or "unknown", model=model or "unknown").inc()
        except Exception:
            pass


def record_ai_provider_switch(from_provider: str, from_model: str, to_provider: str, to_model: str, reason: str) -> None:
    if 'AI_PROVIDER_SWITCH' in globals() and AI_PROVIDER_SWITCH is not None:
        try:
            AI_PROVIDER_SWITCH.labels(
                from_provider=from_provider or "unknown",
                from_model=from_model or "unknown",
                to_provider=to_provider or "unknown",
                to_model=to_model or "unknown",
                reason=reason or "unknown",
            ).inc()
        except Exception:
            pass


def record_ai_provider_switch_success(to_provider: str, to_model: str) -> None:
    if 'AI_PROVIDER_SWITCH_SUCCESS' in globals() and AI_PROVIDER_SWITCH_SUCCESS is not None:
        try:
            AI_PROVIDER_SWITCH_SUCCESS.labels(to_provider=to_provider or "unknown", to_model=to_model or "unknown").inc()
        except Exception:
            pass


def record_ai_provider_switch_failed(to_provider: str, to_model: str, reason: str) -> None:
    if 'AI_PROVIDER_SWITCH_FAILED' in globals() and AI_PROVIDER_SWITCH_FAILED is not None:
        try:
            AI_PROVIDER_SWITCH_FAILED.labels(to_provider=to_provider or "unknown", to_model=to_model or "unknown", reason=reason or "unknown").inc()
        except Exception:
            pass
