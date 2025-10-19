from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.cache import Cache
from utils.persistence import SnapshotStore
from services.stock_data_provider import StockDataProvider
from services.calibration import CalibrationStore, ScoreCalibrator, CalibratorArtifact


logger = get_logger()


@dataclass
class TransformConfig:
    winsorize_lower: float = 0.05
    winsorize_upper: float = 0.95
    standardize: bool = True
    fillna: Optional[float] = None  # None means fill with cross-sectional mean
    industry_neutral: bool = False


@dataclass
class FactorDef:
    id: str
    name: Optional[str] = None
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)
    transform: TransformConfig = field(default_factory=TransformConfig)


class FactorScoringEngine:
    """
    Compute factor values and multi-factor scores with contributions.
    """

    def __init__(
        self,
        data_provider: Optional[StockDataProvider] = None,
        cache: Optional[Cache] = None,
        snapshot_store: Optional[SnapshotStore] = None,
    ):
        self.data_provider = data_provider or StockDataProvider()
        self.cache = cache or Cache(namespace="factor_scores")
        self.snapshot_store = snapshot_store or SnapshotStore()
        self.calibration_store = CalibrationStore()

    # ============ Factor implementations ============
    @staticmethod
    def factor_momentum(df: pd.DataFrame, window: int = 20) -> float:
        if df is None or df.empty or "Close" not in df.columns:
            return np.nan
        if len(df) <= window:
            return np.nan
        try:
            return float(df["Close"].iloc[-1] / df["Close"].iloc[-1 - window] - 1.0)
        except Exception:
            return np.nan

    @staticmethod
    def factor_ma_deviation(df: pd.DataFrame, period: int = 20) -> float:
        if df is None or df.empty or "Close" not in df.columns:
            return np.nan
        if len(df) < period:
            return np.nan
        ma = df["Close"].rolling(window=period).mean()
        last_ma = ma.iloc[-1]
        last_close = df["Close"].iloc[-1]
        if pd.isna(last_ma) or last_ma == 0:
            return np.nan
        return float((last_close - last_ma) / last_ma)

    @staticmethod
    def factor_volatility(df: pd.DataFrame, window: int = 20) -> float:
        if df is None or df.empty or "Close" not in df.columns:
            return np.nan
        if len(df) < window + 1:
            return np.nan
        ret = df["Close"].pct_change()
        vol = ret.rolling(window=window).std().iloc[-1]
        return float(vol) if pd.notna(vol) else np.nan

    @staticmethod
    def compute_factor_by_id(fid: str, df: pd.DataFrame, params: Dict[str, Any]) -> float:
        fid_lower = fid.lower()
        if fid_lower in {"momentum", "mom"}:
            window = int(params.get("window", params.get("period", 20)))
            return FactorScoringEngine.factor_momentum(df, window)
        elif fid_lower in {"ma_dev", "ma_deviation", "mad"}:
            period = int(params.get("period", params.get("window", 20)))
            return FactorScoringEngine.factor_ma_deviation(df, period)
        elif fid_lower in {"vol", "volatility"}:
            window = int(params.get("window", 20))
            return FactorScoringEngine.factor_volatility(df, window)
        else:
            # Unknown factor id
            logger.warning("Unknown factor id '%s' - returning NaN", fid)
            return np.nan

    # ============ Transform utilities ============
    @staticmethod
    def winsorize(s: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
        if s.empty:
            return s
        lower_q, upper_q = s.quantile([lower, upper])
        return s.clip(lower_q, upper_q)

    @staticmethod
    def industry_neutralize(s: pd.Series, industries: Optional[Dict[str, str]]) -> pd.Series:
        if not industries:
            return s
        ind_series = pd.Series({k: industries.get(k, "Unknown") for k in s.index})
        try:
            # Subtract industry group mean
            grouped_mean = s.groupby(ind_series).transform(lambda x: x.mean())
            return s - grouped_mean
        except Exception as e:
            logger.warning("Industry neutralization failed: %s", str(e))
            return s

    @staticmethod
    def standardize_zscore(s: pd.Series) -> pd.Series:
        mean = s.mean()
        std = s.std(ddof=0)
        if std == 0 or np.isclose(std, 0.0) or pd.isna(std):
            return pd.Series([0.0] * len(s), index=s.index)
        return (s - mean) / std

    @staticmethod
    def fill_missing(s: pd.Series, fillna: Optional[float] = None) -> pd.Series:
        if fillna is None:
            return s.fillna(s.mean())
        return s.fillna(fillna)

    # ============ Main scoring ============
    async def score(
        self,
        symbols: List[str],
        factors: List[FactorDef],
        market_type: str = "A",
        window: int = 20,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        industries: Optional[Dict[str, str]] = None,
        page: int = 1,
        page_size: int = 100,
        cache_ttl: int = 300,
    ) -> Dict[str, Any]:
        """
        Compute scores and contributions for a list of symbols.
        """
        # canonical request for caching
        cache_key = self._build_cache_key(
            symbols=symbols,
            factors=factors,
            market_type=market_type,
            window=window,
            start_date=start_date,
            end_date=end_date,
            industries=industries,
            page=page,
            page_size=page_size,
        )

        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # fetch data in batch
        data_map = await self.data_provider.get_multiple_stocks_data(
            stock_codes=symbols,
            market_type=market_type,
            start_date=start_date,
            end_date=end_date,
            max_concurrency=10,
        )

        # compute cross-sectional factor values per factor id
        factor_values: Dict[str, pd.Series] = {}
        for f in factors:
            values: Dict[str, float] = {}
            for sym in symbols:
                df = data_map.get(sym)
                val = self.compute_factor_by_id(f.id, df, {**f.params, "window": f.params.get("window", window)})
                values[sym] = val
            s = pd.Series(values)

            # apply transforms
            s = self.winsorize(s, f.transform.winsorize_lower, f.transform.winsorize_upper)
            s = self.fill_missing(s, f.transform.fillna)
            if f.transform.industry_neutral and industries:
                s = self.industry_neutralize(s, industries)
            z = self.standardize_zscore(s) if f.transform.standardize else s

            factor_values[f.id] = z

        # compose final scores and contribs
        results: List[Dict[str, Any]] = []
        for sym in symbols:
            contribs = []
            total = 0.0
            for f in factors:
                z = float(factor_values[f.id].get(sym, np.nan))
                if np.isnan(z):
                    z = 0.0
                contrib = float(f.weight) * z
                contribs.append({
                    "factor_id": f.id,
                    "factor_name": f.name or f.id,
                    "weight": float(f.weight),
                    "z": z,
                    "contrib": contrib,
                })
                total += contrib
            results.append({
                "symbol": sym,
                "total_score": float(total),
                "contribs": contribs,
            })

        # compute cross-sectional quantiles and calibrated probabilities
        try:
            totals = np.array([r["total_score"] for r in results], dtype=float)
            n = len(totals)
            if n > 0:
                sorted_totals = np.sort(totals)
                left = np.searchsorted(sorted_totals, totals, side="left").astype(float)
                right = np.searchsorted(sorted_totals, totals, side="right").astype(float)
                # average rank method
                quantiles = (left + right) / 2.0 / float(n)
            else:
                quantiles = np.zeros(0, dtype=float)

            # try to load existing calibration artifact
            calib_key = self._build_calibration_key(factors=factors, market_type=market_type, window=window)
            artifact = self.calibration_store.load(calib_key)
            calibrated = None
            if artifact is not None:
                try:
                    valid_until = datetime.fromisoformat(artifact.valid_until)
                    if valid_until < datetime.utcnow():
                        artifact = None
                except Exception:
                    artifact = None
            if artifact is not None and len(totals) > 0:
                calibrator = ScoreCalibrator(method=artifact.method)
                calibrator.load(artifact)
                calibrated = calibrator.predict(totals)
            # attach metrics to results
            for i, r in enumerate(results):
                r["quantile"] = float(quantiles[i]) if len(quantiles) > i else 0.0
                if calibrated is not None and len(calibrated) > i:
                    r["calibrated_prob"] = float(calibrated[i])
                else:
                    r["calibrated_prob"] = float(quantiles[i]) if len(quantiles) > i else 0.0
        except Exception as e:
            logger.warning("Failed to enrich results with calibration info: %s", str(e))

        # sort by total_score desc
        results.sort(key=lambda x: x["total_score"], reverse=True)

        # pagination
        total_count = len(results)
        start = max((page - 1) * page_size, 0)
        end_idx = start + page_size
        paged_results = results[start:end_idx]

        response = {
            "total": total_count,
            "page": page,
            "page_size": page_size,
            "results": paged_results,
        }

        # persist snapshot (one row per factor per symbol)
        try:
            self._persist_snapshot(
                factors=factors,
                factor_values=factor_values,
                symbols=symbols,
                market_type=market_type,
                window=window,
            )
        except Exception as e:
            logger.warning("Snapshot persistence failed: %s", str(e))

        # cache
        self.cache.set(cache_key, response, ttl_seconds=cache_ttl)

        return response

    def _persist_snapshot(
        self,
        factors: List[FactorDef],
        factor_values: Dict[str, pd.Series],
        symbols: List[str],
        market_type: str,
        window: int,
    ) -> None:
        ts = pd.Timestamp.utcnow()
        request_id = hashlib.md5((str(ts.value) + json.dumps([f.id for f in factors])).encode("utf-8")).hexdigest()

        rows = []
        for f in factors:
            zseries = factor_values.get(f.id, pd.Series(dtype=float))
            for sym in symbols:
                z = float(zseries.get(sym, np.nan))
                if np.isnan(z):
                    z = 0.0
                rows.append({
                    "ts": ts,
                    "request_id": request_id,
                    "symbol": sym,
                    "factor_id": f.id,
                    "factor_name": f.name or f.id,
                    "value": np.nan,  # raw value not retained after xform; keep NaN for now
                    "z": z,
                    "weight": float(f.weight),
                    "contrib": float(f.weight) * z,
                    "window": int(window),
                    "market_type": market_type,
                })
        df = pd.DataFrame(rows)
        self.snapshot_store.save(df)

    @staticmethod
    def _build_cache_key(
        symbols: List[str],
        factors: List[FactorDef],
        market_type: str,
        window: int,
        start_date: Optional[str],
        end_date: Optional[str],
        industries: Optional[Dict[str, str]],
        page: int,
        page_size: int,
    ) -> str:
        payload = {
            "symbols": symbols,
            "factors": [
                {
                    "id": f.id,
                    "name": f.name,
                    "weight": f.weight,
                    "params": f.params,
                    "transform": {
                        "winsorize_lower": f.transform.winsorize_lower,
                        "winsorize_upper": f.transform.winsorize_upper,
                        "standardize": f.transform.standardize,
                        "fillna": f.transform.fillna,
                        "industry_neutral": f.transform.industry_neutral,
                    },
                }
                for f in factors
            ],
            "market_type": market_type,
            "window": window,
            "start_date": start_date,
            "end_date": end_date,
            "industries": industries,
            "page": page,
            "page_size": page_size,
        }
        m = hashlib.md5(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return m

    @staticmethod
    def _build_calibration_key(
        factors: List[FactorDef],
        market_type: str,
        window: int,
    ) -> str:
        payload = {
            "factors": [
                {
                    "id": f.id,
                    "name": f.name,
                    "weight": f.weight,
                    "params": f.params,
                    "transform": {
                        "winsorize_lower": f.transform.winsorize_lower,
                        "winsorize_upper": f.transform.winsorize_upper,
                        "standardize": f.transform.standardize,
                        "fillna": f.transform.fillna,
                        "industry_neutral": f.transform.industry_neutral,
                    },
                }
                for f in factors
            ],
            "market_type": market_type,
            "window": window,
        }
        return "calib_" + hashlib.md5(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
