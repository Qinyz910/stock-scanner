import pytest

pytest.skip(
    "Manual integration test requiring external market data", allow_module_level=True
)

import akshare as ak

print(f"akshare version: {ak.__version__}")
# df = ak.stock_zh_a_hist(symbol="000858", 
#                         start_date="20250301", 
#                         end_date="20250310", 
#                         adjust="qfq")
# print(df)

stock_us_daily_df = ak.stock_us_daily(symbol="AAPL", adjust="qfq")

print(stock_us_daily_df)
