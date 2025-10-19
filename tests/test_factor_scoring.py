import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from services.factor_scoring import (
    FactorScoringEngine,
    FactorDef,
    TransformConfig,
)


def make_price_df(prices):
    idx = pd.date_range(end=pd.Timestamp.today(), periods=len(prices), freq="D")
    return pd.DataFrame({"Close": prices}, index=idx)


def test_transforms_and_zscore():
    s = pd.Series({"A": -100, "B": 0, "C": 10, "D": 200, "E": 30}, dtype=float)

    # winsorize
    s_w = FactorScoringEngine.winsorize(s, 0.05, 0.95)
    assert s_w.max() <= s.quantile(0.95) + 1e-12
    assert s_w.min() >= s.quantile(0.05) - 1e-12

    # fill missing
    s2 = s.copy()
    s2.loc["B"] = np.nan
    s2_f = FactorScoringEngine.fill_missing(s2)
    assert not s2_f.isna().any()

    # standardize
    z = FactorScoringEngine.standardize_zscore(s_w)
    assert abs(z.mean()) < 1e-8
    # std may deviate slightly due to ddof=0
    assert 0.9 < z.std(ddof=0) < 1.1


def test_factor_impls():
    df = make_price_df([10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15])
    mom = FactorScoringEngine.factor_momentum(df, window=5)
    mad = FactorScoringEngine.factor_ma_deviation(df, period=5)
    vol = FactorScoringEngine.factor_volatility(df, window=5)

    assert not np.isnan(mom)
    assert not np.isnan(mad)
    assert not np.isnan(vol)


def test_contribution_decomposition_end_to_end():
    # Prepare dummy data provider
    class DummyProvider:
        async def get_multiple_stocks_data(self, stock_codes, market_type, start_date, end_date, max_concurrency=10):
            base = [10, 10, 10, 10, 10, 10, 10, 10, 10]
            data = {}
            for i, sym in enumerate(stock_codes):
                # create slight differences among symbols
                prices = [p + i for p in base]
                data[sym] = make_price_df(prices)
            return data

    engine = FactorScoringEngine(data_provider=DummyProvider())

    factors = [
        FactorDef(id="momentum", name="Momentum", weight=0.6, params={"window": 3}),
        FactorDef(id="volatility", name="Vol", weight=0.4, params={"window": 3}),
    ]

    symbols = ["AAA", "BBB", "CCC"]

    res = asyncio.run(
        engine.score(
            symbols=symbols,
            factors=factors,
            market_type="A",
            window=3,
            industries={"AAA": "I1", "BBB": "I1", "CCC": "I2"},
            page=1,
            page_size=10,
        )
    )

    assert "results" in res and len(res["results"]) == len(symbols)

    for item in res["results"]:
        total = item["total_score"]
        contrib_sum = sum(c["contrib"] for c in item["contribs"])
        assert abs(total - contrib_sum) < 1e-9
