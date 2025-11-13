from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger
from services.stock_data_provider import StockDataProvider

logger = get_logger()


@dataclass
class BacktestParams:
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    fee_bps: float = 1.0  # 1 bps per trade
    slippage_bps: float = 1.0
    initial_cash: float = 100000.0


def _to_weights(symbols: List[str], method: str = "equal", data: Dict[str, pd.DataFrame] | None = None) -> Dict[str, float]:
    n = max(1, len(symbols))
    if method == "equal" or data is None:
        return {s: 1.0 / n for s in symbols}
    if method == "vol_inverse":
        vols = {}
        for s in symbols:
            df = data.get(s, pd.DataFrame()) if data else pd.DataFrame()
            r = df.get("Close", pd.Series(dtype=float)).pct_change().dropna()
            vol = float(r.std(ddof=0)) if len(r) > 0 else np.nan
            vols[s] = vol
        inv = {s: (1.0 / v if np.isfinite(v) and v > 0 else 0.0) for s, v in vols.items()}
        tot = sum(inv.values()) or 1.0
        return {s: (w / tot) for s, w in inv.items()}
    return {s: 1.0 / n for s in symbols}


def run_backtest(symbols: List[str], market_type: str, params: BacktestParams, provider: Optional[StockDataProvider] = None, progress_cb=None, as_of: Optional[datetime] = None) -> dict:
    """
    Run backtest with point-in-time data consistency.
    
    Args:
        as_of: Point-in-time timestamp - ensures data is truncated to avoid look-ahead bias
    """
    provider = provider or StockDataProvider()
    progress_cb = progress_cb or (lambda pct, msg="": None)

    # fetch data with point-in-time constraint to match online scoring
    progress_cb(5, "fetching data")
    import asyncio
    from datetime import datetime as dt
    data_map = asyncio.run(provider.get_multiple_stocks_data(symbols, market_type=market_type, start_date=params.start_date, end_date=params.end_date, max_concurrency=8, as_of=as_of))

    # align dates
    progress_cb(20, "aligning data")
    closes = []
    for s in symbols:
        df = data_map.get(s)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        c = df["Close"].astype(float).rename(s)
        closes.append(c)
    if not closes:
        raise ValueError("no valid data")
    px = pd.concat(closes, axis=1).dropna()

    # weights
    w = _to_weights(symbols=list(px.columns), method="equal", data=data_map)
    wv = pd.Series(w)

    # returns
    progress_cb(40, "computing returns")
    ret = px.pct_change().fillna(0.0)
    port_ret = (ret * wv).sum(axis=1)

    # simple buy-and-hold equity
    progress_cb(70, "simulating")
    equity = (1.0 + port_ret).cumprod() * params.initial_cash
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0, np.nan)

    # metrics
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    vol = float(port_ret.std(ddof=0) * np.sqrt(252)) if len(port_ret) > 1 else 0.0
    sharpe = float(port_ret.mean() / (port_ret.std(ddof=0) + 1e-12) * np.sqrt(252)) if len(port_ret) > 1 else 0.0
    maxdd = float(dd.max()) if len(dd) else 0.0

    progress_cb(95, "finalizing")
    out = {
        "symbols": list(px.columns),
        "start": str(px.index[0].date()),
        "end": str(px.index[-1].date()),
        "equity_curve": {
            "dates": [str(d.date()) for d in equity.index],
            "equity": [float(x) for x in equity.values],
        },
        "metrics": {
            "total_return": total_ret,
            "sharpe": sharpe,
            "max_drawdown": maxdd,
            "volatility": vol,
            "turnover": 0.0,
        },
        "logs": [],
    }
    progress_cb(100, "done")
    return out
