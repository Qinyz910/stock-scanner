from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


def _safe_pct_change(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    try:
        r = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        return r
    except Exception:
        return pd.Series(dtype=float)


def rolling_volatility(close: pd.Series, window: int = 20) -> float:
    r = _safe_pct_change(close)
    if len(r) < max(2, window):
        return float("nan")
    vol = r.iloc[-window:].std(ddof=0)
    return float(vol) if pd.notna(vol) else float("nan")


def mean_return(close: pd.Series, window: int = 20) -> float:
    r = _safe_pct_change(close)
    if len(r) < 1:
        return float("nan")
    if window and len(r) >= window:
        r = r.iloc[-window:]
    return float(r.mean()) if len(r) else float("nan")


def max_drawdown(close: pd.Series, window: Optional[int] = None) -> float:
    if close is None or len(close) == 0:
        return float("nan")
    s = close
    if window and len(s) >= window:
        s = close.iloc[-window:]
    # compute drawdown as percentage
    cummax = s.cummax()
    drawdown = (cummax - s) / cummax.replace(0, np.nan)
    dd = drawdown.max()
    return float(dd) if pd.notna(dd) else float("nan")


def calmar_ratio(close: pd.Series, window: int = 20) -> float:
    mr = mean_return(close, window)
    dd = max_drawdown(close, window)
    if not np.isfinite(dd) or dd <= 0:
        return float("inf")
    return float(mr / dd)


def risk_adjusted_score(close: pd.Series, window: int = 20) -> float:
    mr = mean_return(close, window)
    vol = rolling_volatility(close, window)
    if not np.isfinite(vol) or vol <= 0:
        return 0.0
    return float(mr / vol)


def bootstrap_mean_ci(returns: pd.Series, n_boot: int = 300, ci: float = 0.95) -> Tuple[float, float]:
    """Bootstrap confidence interval for mean of returns.

    Returns (low, high). If not enough data, returns (nan, nan).
    """
    r = pd.Series(returns).dropna().values
    if r.size < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(42)
    n = r.size
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = r[rng.integers(0, n, size=n)]
        means[i] = sample.mean()
    alpha = (1 - ci) / 2
    low = float(np.quantile(means, alpha))
    high = float(np.quantile(means, 1 - alpha))
    return low, high


def confidence_from_ci(width: float, reference: float) -> float:
    """Convert CI width and reference scale to a [0,1] confidence heuristic.

    If reference is near zero, fall back to damping by width only.
    """
    if not np.isfinite(width) or width <= 0:
        return 0.9
    ref = abs(reference)
    if ref <= 1e-8:
        return float(1.0 / (1.0 + width * 50.0))
    ratio = width / (ref + 1e-12)
    # larger ratio -> lower confidence
    conf = 1.0 / (1.0 + ratio)
    # clamp
    return float(max(0.0, min(1.0, conf)))
