from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class Factor:
    id: str
    compute: Callable[[pd.DataFrame, Dict[str, Any]], pd.Series | pd.DataFrame]
    description: str = ""


def _sma(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    period = int(params.get("period", params.get("window", 20)))
    return df["Close"].rolling(window=period).mean()


def _ema(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    period = int(params.get("period", params.get("window", 20)))
    return df["Close"].ewm(span=period, adjust=False).mean()


def _rsi(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    period = int(params.get("period", 14))
    close = df["Close"].astype(float)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss.replace(0.0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    fast = int(params.get("fast", 12))
    slow = int(params.get("slow", 26))
    signal = int(params.get("signal", 9))
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    out = pd.DataFrame({"macd": macd, "signal": sig, "hist": hist})
    return out


def _atr(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    period = int(params.get("period", 14))
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def _bbands(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    period = int(params.get("period", 20))
    num_std = float(params.get("std", params.get("num_std", 2)))
    mid = df["Close"].rolling(window=period).mean()
    sd = df["Close"].rolling(window=period).std(ddof=0)
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    return pd.DataFrame({"middle": mid, "upper": upper, "lower": lower})


class FactorRegistry:
    def __init__(self):
        self._factors: Dict[str, Factor] = {}
        self.register("sma", _sma, "Simple moving average")
        self.register("ema", _ema, "Exponential moving average")
        self.register("rsi", _rsi, "Relative Strength Index")
        self.register("macd", _macd, "MACD components")
        self.register("atr", _atr, "Average True Range")
        self.register("bbands", _bbands, "Bollinger Bands")

    def register(self, fid: str, func: Callable[[pd.DataFrame, Dict[str, Any]], pd.Series | pd.DataFrame], desc: str = "") -> None:
        self._factors[fid.lower()] = Factor(id=fid.lower(), compute=func, description=desc)

    def compute(self, fid: str, df: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series | pd.DataFrame:
        f = self._factors.get(fid.lower())
        if f is None:
            raise ValueError(f"Unknown factor id: {fid}")
        return f.compute(df, params or {})

    def list(self) -> Dict[str, str]:
        return {k: v.description for k, v in self._factors.items()}


# Singleton registry
REGISTRY = FactorRegistry()
