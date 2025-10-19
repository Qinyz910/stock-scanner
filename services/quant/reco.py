from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Recommendation:
    symbol: str
    rating: str
    score: float
    confidence: float
    plan: Dict[str, float]
    risks: List[str]
    drivers: List[str]


def aggregate_score_to_rating(score: float) -> str:
    if score >= 1.5:
        return "strong_buy"
    if score >= 0.5:
        return "buy"
    if score > -0.5:
        return "neutral"
    if score > -1.5:
        return "sell"
    return "strong_sell"


def recommend(symbols: List[str], scores: Dict[str, float], risk_tag: Optional[Dict[str, str]] = None,
              risk_appetite: str = "balanced") -> List[Recommendation]:
    out: List[Recommendation] = []
    risk_tag = risk_tag or {}
    for s in symbols:
        sc = float(scores.get(s, 0.0))
        # confidence heuristic: dampen by risk
        risk = risk_tag.get(s, "low")
        conf = 0.7
        if risk == "high":
            conf *= 0.6
        elif risk == "medium":
            conf *= 0.8
        rating = aggregate_score_to_rating(sc)
        # simple plan: stop-loss/take-profit based on score
        tp = 0.05 if sc >= 0 else 0.02
        sl = 0.02 if sc >= 0 else 0.05
        if risk_appetite == "aggressive":
            tp *= 1.5
            sl *= 1.3
        elif risk_appetite == "conservative":
            tp *= 0.7
            sl *= 0.7
        plan = {
            "entry_offset_pct": 0.0,
            "take_profit_pct": tp,
            "stop_loss_pct": sl,
            "horizon_days": 20,
        }
        risks = [f"market_volatility:{risk}"]
        drivers = ["multi_factor_score"]
        out.append(Recommendation(symbol=s, rating=rating, score=sc, confidence=conf, plan=plan, risks=risks, drivers=drivers))
    return out
