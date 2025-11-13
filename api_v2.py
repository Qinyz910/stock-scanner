from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from utils.cache import Cache
from services.stock_data_provider import StockDataProvider
from services.quant.factors import REGISTRY as FACTORS_REG
from services.quant.backtest import run_backtest, BacktestParams
from utils.task_queue import GLOBAL_TASK_QUEUE
from services.quant.reco import recommend as reco_recommend
from utils.exceptions import ValidationError as AppValidationError, NotFoundError

api_v2_router = APIRouter(prefix="/api/v2", tags=["v2"])


# --------- Models ---------
class FactorSpec(BaseModel):
    id: str
    params: Dict[str, Any] = Field(default_factory=dict)
    output: str = Field(default="last")  # last | series


class FactorComputeRequest(BaseModel):
    symbols: List[str]
    market: str = "A"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    factors: List[FactorSpec]


class BacktestRunRequest(BaseModel):
    symbols: List[str]
    market: str = "A"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class PortfolioOptimizeRequest(BaseModel):
    symbols: List[str]
    method: str = Field(default="equal", description="equal|vol_inverse")


class MLPredictRequest(BaseModel):
    symbols: List[str]
    market: str = "A"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    model_version: Optional[str] = None


class RecommendRequest(BaseModel):
    symbols: List[str]
    risk_appetite: str = Field(default="balanced")
    scores: Optional[Dict[str, float]] = None


provider = StockDataProvider()
cache_factors = Cache(namespace="factors_v2")


# --------- Health ---------
@api_v2_router.get("/health/cache")
async def cache_health():
    return cache_factors.health()


# --------- Factor endpoints ---------
@api_v2_router.get("/factors")
async def list_factors():
    return {"factors": FACTORS_REG.list()}


@api_v2_router.post("/factors/compute")
async def factors_compute(req: FactorComputeRequest):
    if not req.symbols:
        raise AppValidationError("Symbols are required")
    if not req.factors:
        raise AppValidationError("Factors are required")

    # Cache key based on payload
    payload = req.model_dump()
    cache_key = f"compute:{json.dumps(payload, sort_keys=True, ensure_ascii=False)}"
    cached = cache_factors.get(cache_key)
    if cached is not None:
        return cached

    data_map = await provider.get_multiple_stocks_data(
        stock_codes=req.symbols,
        market_type=req.market,
        start_date=req.start_date,
        end_date=req.end_date,
        max_concurrency=8,
    )

    results: List[Dict[str, Any]] = []
    for sym in req.symbols:
        df = data_map.get(sym)
        if df is None or df.empty:
            results.append({"symbol": sym, "factors": {}, "error": "no_data"})
            continue
        out: Dict[str, Any] = {}
        for fs in req.factors:
            try:
                arr = FACTORS_REG.compute(fs.id, df, fs.params)
                if isinstance(arr, pd.DataFrame):
                    if fs.output == "series":
                        out[fs.id] = {col: arr[col].dropna().tolist() for col in arr.columns}
                    else:
                        out[fs.id] = {col: float(arr[col].dropna().iloc[-1]) if arr[col].notna().any() else None for col in arr.columns}
                else:
                    if fs.output == "series":
                        out[fs.id] = arr.dropna().tolist()
                    else:
                        out[fs.id] = float(arr.dropna().iloc[-1]) if arr.notna().any() else None
            except Exception as e:
                out[fs.id] = None
        results.append({"symbol": sym, "factors": out})

    resp = {"results": results}
    cache_factors.set(cache_key, resp, ttl_seconds=300)
    return resp


# --------- Backtest endpoints ---------
@api_v2_router.post("/backtest/run")
async def backtest_run(req: BacktestRunRequest):
    p = BacktestParams(
        start_date=req.start_date,
        end_date=req.end_date,
        fee_bps=float(req.params.get("fee_bps", 1.0)),
        slippage_bps=float(req.params.get("slippage_bps", 1.0)),
        initial_cash=float(req.params.get("initial_cash", 100000.0)),
    )

    def job(progress_cb):
        return run_backtest(req.symbols, req.market, p, provider=provider, progress_cb=progress_cb)

    job_id = GLOBAL_TASK_QUEUE.submit(job)
    return {"job_id": job_id}


@api_v2_router.get("/backtest/{job_id}")
async def backtest_status(job_id: str):
    job = GLOBAL_TASK_QUEUE.get(job_id)
    if job is None:
        raise NotFoundError("Backtest job not found")
    return job.to_dict()


@api_v2_router.get("/backtest/{job_id}/stream")
async def backtest_stream(job_id: str):
    async def gen():
        last_len = 0
        while True:
            job = GLOBAL_TASK_QUEUE.get(job_id)
            if job is None:
                yield json.dumps({"event": "error", "message": "job not found"}) + "\n"
                return
            snapshot = job.to_dict()
            logs = snapshot.get("logs", [])
            if len(logs) > last_len:
                for line in logs[last_len:]:
                    yield json.dumps({"event": "log", "data": line}) + "\n"
                last_len = len(logs)
            yield json.dumps({"event": "progress", "progress": snapshot.get("progress", 0), "status": snapshot.get("status")}) + "\n"
            if snapshot.get("status") in {"done", "error"}:
                yield json.dumps({"event": "end", "status": snapshot.get("status"), "result": snapshot.get("result"), "error": snapshot.get("error")}) + "\n"
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="application/json")


# --------- Portfolio ---------
@api_v2_router.post("/portfolio/optimize")
async def portfolio_optimize(req: PortfolioOptimizeRequest):
    # simple equal weight or volatility inverse placeholder
    if not req.symbols:
        raise AppValidationError("Symbols are required")
    weights = {s: 1.0 / len(req.symbols) for s in req.symbols}
    return {"weights": weights, "method": req.method}


# --------- ML ---------
@api_v2_router.post("/ml/predict")
async def ml_predict(req: MLPredictRequest):
    data_map = await provider.get_multiple_stocks_data(req.symbols, req.market, req.start_date, req.end_date, max_concurrency=8)
    window = 5
    scores = {}
    for s in req.symbols:
        df = data_map.get(s)
        if df is None or df.empty or "Close" not in df.columns:
            scores[s] = 0.0
            continue
        try:
            close = df["Close"].astype(float)
            if len(close) > window:
                scores[s] = float(close.iloc[-1] / close.iloc[-1 - window] - 1.0)
            else:
                scores[s] = 0.0
        except Exception:
            scores[s] = 0.0
    return {"model_version": req.model_version or "baseline_mom", "scores": scores}


# --------- Recommendation ---------
@api_v2_router.post("/signals/recommend")
async def recommend(req: RecommendRequest):
    symbols = req.symbols
    if not symbols:
        raise AppValidationError("Symbols are required")
    scores = req.scores or {s: 0.0 for s in symbols}
    recos = reco_recommend(symbols, scores, risk_tag=None, risk_appetite=req.risk_appetite)
    return {"results": [r.__dict__ for r in recos]}
