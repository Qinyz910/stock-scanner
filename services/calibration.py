from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger()


# -------------------- Metrics --------------------

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


# -------------------- Reliability curve --------------------

def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
    - bin_centers: representative probability per bin (mean of predicted prob)
    - frac_positives: empirical positive rate per bin
    - counts: number of samples per bin
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) != len(y_prob) or len(y_true) == 0:
        return np.array([]), np.array([]), np.array([])

    # Equal-frequency binning on predicted probabilities
    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]
    n = len(y_true)
    bins = np.linspace(0, n, n_bins + 1, dtype=int)
    centers = []
    fracs = []
    counts = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if hi <= lo:
            continue
        yt = y_true_sorted[lo:hi]
        yp = y_prob_sorted[lo:hi]
        centers.append(float(np.mean(yp)))
        fracs.append(float(np.mean(yt)))
        counts.append(int(len(yt)))
    return np.asarray(centers), np.asarray(fracs), np.asarray(counts)


# -------------------- Calibrators --------------------

@dataclass
class CalibratorArtifact:
    method: str
    created_at: str
    valid_until: str
    meta: Dict[str, Any]
    # representation for mapping
    thresholds: Optional[List[float]] = None  # for isotonic: sorted thresholds of scores
    values: Optional[List[float]] = None      # for isotonic: mapped probabilities per threshold
    bin_edges: Optional[List[float]] = None   # for quantile-binning: score edges (len = n_bins+1)
    bin_probs: Optional[List[float]] = None   # for quantile-binning: empirical probs per bin (len = n_bins)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "CalibratorArtifact":
        d = json.loads(s)
        return CalibratorArtifact(**d)


class ScoreCalibrator:
    """
    Monotonic calibrator mapping scores -> probabilities using isotonic regression (PAV)
    or equal-frequency binning.
    """

    def __init__(self, method: str = "isotonic", n_bins: int = 10):
        method = (method or "isotonic").lower()
        if method not in {"isotonic", "quantile"}:
            method = "quantile"
        self.method = method
        self.n_bins = int(max(2, n_bins))
        self._artifact: Optional[CalibratorArtifact] = None

    @staticmethod
    def _pav_isotonic(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pool Adjacent Violators algorithm for isotonic regression (non-decreasing).
        Returns thresholds (sorted scores) and fitted values (piecewise-constant probs).
        """
        order = np.argsort(scores)
        x = scores[order]
        y = labels[order].astype(float)
        # Initialize blocks
        values = y.copy()
        weights = np.ones_like(values)
        # Each block is represented implicitly via values + weights; perform PAV
        i = 0
        while i < len(values) - 1:
            if values[i] <= values[i + 1] + 1e-12:
                i += 1
                continue
            # merge blocks j..i where monotonicity is violated backward
            j = i
            while j >= 0 and values[j] > values[j + 1] + 1e-12:
                w = weights[j] + weights[j + 1]
                v = (weights[j] * values[j] + weights[j + 1] * values[j + 1]) / w
                values[j] = v
                weights[j] = w
                # remove block j+1 by shifting left
                values = np.delete(values, j + 1)
                weights = np.delete(weights, j + 1)
                x = np.delete(x, j + 1)
                j -= 1
            i = max(j, 0)
        # Now expand piecewise-constant mapping back across scores
        # Produce thresholds at block boundaries and corresponding values
        thresholds = []
        fitted = []
        # reconstruct from cumulative sizes in weights; need original sorted scores to determine boundaries
        pos = 0
        for w, v in zip(weights, values):
            # use the max score inside this block as threshold
            idx_hi = min(len(order) - 1, pos + int(w) - 1)
            thresholds.append(float(x[idx_hi]))
            fitted.append(float(max(0.0, min(1.0, v))))
            pos += int(w)
        return np.asarray(thresholds), np.asarray(fitted)

    @staticmethod
    def _quantile_binning(scores: np.ndarray, labels: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
        order = np.argsort(scores)
        s_sorted = scores[order]
        y_sorted = labels[order].astype(float)
        n = len(scores)
        edges_idx = np.linspace(0, n, n_bins + 1, dtype=int)
        edges = []
        probs = []
        for i in range(n_bins):
            lo, hi = edges_idx[i], edges_idx[i + 1]
            if hi <= lo:
                continue
            seg_scores = s_sorted[lo:hi]
            seg_labels = y_sorted[lo:hi]
            edges.append(float(seg_scores[-1]))
            probs.append(float(np.mean(seg_labels)))
        # Convert to arrays: edges length = n_bins (last edge at each bin's high score)
        return np.asarray(edges), np.asarray(probs)

    def fit(self, scores: np.ndarray, labels: np.ndarray, meta: Optional[Dict[str, Any]] = None,
            valid_days: int = 7) -> CalibratorArtifact:
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if len(scores) == 0:
            raise ValueError("No calibration data")
        if self.method == "isotonic":
            thresholds, values = self._pav_isotonic(scores, labels)
            artifact = CalibratorArtifact(
                method="isotonic",
                created_at=datetime.utcnow().isoformat(),
                valid_until=(datetime.utcnow() + timedelta(days=int(valid_days))).isoformat(),
                meta=meta or {},
                thresholds=thresholds.tolist(),
                values=values.tolist(),
            )
        else:
            bin_edges, bin_probs = self._quantile_binning(scores, labels, self.n_bins)
            artifact = CalibratorArtifact(
                method="quantile",
                created_at=datetime.utcnow().isoformat(),
                valid_until=(datetime.utcnow() + timedelta(days=int(valid_days))).isoformat(),
                meta=meta or {},
                bin_edges=bin_edges.tolist(),
                bin_probs=bin_probs.tolist(),
            )
        self._artifact = artifact
        return artifact

    def is_fitted(self) -> bool:
        return self._artifact is not None

    def load(self, artifact: CalibratorArtifact) -> None:
        self._artifact = artifact
        self.method = artifact.method

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self._artifact is None:
            raise RuntimeError("Calibrator is not fitted")
        s = np.asarray(scores, dtype=float)
        if self._artifact.method == "isotonic":
            thr = np.asarray(self._artifact.thresholds, dtype=float)
            vals = np.asarray(self._artifact.values, dtype=float)
            # For each score, find first threshold >= score, mapping to that block value
            idx = np.searchsorted(thr, s, side="right")
            idx = np.clip(idx, 0, len(vals) - 1)
            return vals[idx]
        else:
            edges = np.asarray(self._artifact.bin_edges, dtype=float)
            probs = np.asarray(self._artifact.bin_probs, dtype=float)
            if len(edges) == 0:
                return np.full_like(s, fill_value=float(np.mean(probs)) if len(probs) else 0.5, dtype=float)
            idx = np.searchsorted(edges, s, side="right") - 1
            idx = np.clip(idx, 0, len(probs) - 1)
            return probs[idx]


# -------------------- Persistence --------------------

class CalibrationStore:
    def __init__(self, base_dir: str = "data/calibration"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        safe = key.replace("/", "_")
        return os.path.join(self.base_dir, f"{safe}.json")

    def save(self, key: str, artifact: CalibratorArtifact) -> str:
        path = self._path(key)
        with open(path, "w", encoding="utf-8") as f:
            f.write(artifact.to_json())
        logger.info("Saved calibration artifact at %s", path)
        return path

    def load(self, key: str) -> Optional[CalibratorArtifact]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = f.read()
            art = CalibratorArtifact.from_json(data)
            return art
        except Exception as e:
            logger.warning("Failed to load calibration artifact %s: %s", path, str(e))
            return None


# -------------------- Dataset building (optional, simple) --------------------

class CalibrationDatasetBuilder:
    """
    Build calibration dataset by historical replay: score -> next period excess return > 0
    Note: This simple implementation loops per date and symbol; it is designed to be used
    on small universes or short windows.
    """

    def __init__(self, data_provider, factor_engine):
        self.data_provider = data_provider
        self.factor_engine = factor_engine

    async def build(
        self,
        symbols: List[str],
        factors: List[Any],
        market_type: str = "A",
        window: int = 20,
        horizon: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_dates: int = 60,
        industries: Optional[Dict[str, str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns arrays (scores, labels)
        label = 1 if next horizon return beats cross-sectional median on that date.
        """
        # fetch data once
        data_map = await self.data_provider.get_multiple_stocks_data(
            stock_codes=symbols,
            market_type=market_type,
            start_date=start_date,
            end_date=end_date,
            max_concurrency=10,
        )
        # Determine dates intersection
        all_indices = [df.index for df in data_map.values() if df is not None and not df.empty]
        if not all_indices:
            return np.array([]), np.array([])
        common_index = sorted(list(set().union(*[set(idx) for idx in all_indices])))
        if len(common_index) == 0:
            return np.array([]), np.array([])
        # limit number of dates to evaluate to keep performance reasonable
        if max_dates and len(common_index) > max_dates:
            # take evenly spaced dates across index
            step = max(1, len(common_index) // max_dates)
            eval_dates = common_index[window:- (horizon or window) : step]
        else:
            eval_dates = common_index[window: - (horizon or window)]
        if not eval_dates:
            return np.array([]), np.array([])
        scores_list: List[float] = []
        labels_list: List[int] = []
        H = int(horizon or window)
        # Precompute closes for returns
        closes_map: Dict[str, pd.Series] = {}
        for sym, df in data_map.items():
            if df is None or df.empty or "Close" not in df.columns:
                continue
            closes_map[sym] = df["Close"].copy()
        for dt in eval_dates:
            # compute factor z for each symbol at dt
            zmap: Dict[str, float] = {}
            # compute cross-sectional z per factor then sum weighted contributions
            contribs_per_sym: Dict[str, float] = {sym: 0.0 for sym in symbols}
            for f in factors:
                raw_vals = {}
                for sym in symbols:
                    df = data_map.get(sym)
                    if df is None or df.empty:
                        raw_vals[sym] = np.nan
                        continue
                    # slice up to dt
                    df_slice = df[df.index <= dt]
                    val = self.factor_engine.compute_factor_by_id(f.id, df_slice, {**f.params, "window": f.params.get("window", window)})
                    raw_vals[sym] = val
                s = pd.Series(raw_vals)
                # transforms (cross-sectional)
                s = self.factor_engine.winsorize(s, f.transform.winsorize_lower, f.transform.winsorize_upper)
                s = self.factor_engine.fill_missing(s, f.transform.fillna)
                if getattr(f.transform, "industry_neutral", False) and industries:
                    s = self.factor_engine.industry_neutralize(s, industries)
                z = self.factor_engine.standardize_zscore(s) if getattr(f.transform, "standardize", True) else s
                for sym in symbols:
                    zval = float(z.get(sym, np.nan))
                    if np.isnan(zval):
                        zval = 0.0
                    contribs_per_sym[sym] = contribs_per_sym.get(sym, 0.0) + float(f.weight) * zval
            # Now we have cross-sectional scores at dt
            # Compute next horizon returns and label
            rets = {}
            for sym in symbols:
                ser = closes_map.get(sym)
                if ser is None or len(ser) == 0:
                    rets[sym] = np.nan
                    continue
                try:
                    # find t index
                    if dt not in ser.index:
                        rets[sym] = np.nan
                        continue
                    pos = ser.index.get_loc(dt)
                    if isinstance(pos, slice):
                        pos = pos.stop - 1
                    next_pos = pos + H
                    if next_pos >= len(ser):
                        rets[sym] = np.nan
                        continue
                    r = float(ser.iloc[next_pos] / ser.iloc[pos] - 1.0)
                    rets[sym] = r
                except Exception:
                    rets[sym] = np.nan
            r_series = pd.Series(rets).dropna()
            if r_series.empty:
                continue
            median_ret = float(r_series.median())
            for sym in r_series.index:
                score = contribs_per_sym.get(sym, 0.0)
                label = int(r_series[sym] - median_ret > 0)
                scores_list.append(score)
                labels_list.append(label)
        if not scores_list:
            return np.array([]), np.array([])
        return np.asarray(scores_list, dtype=float), np.asarray(labels_list, dtype=int)
