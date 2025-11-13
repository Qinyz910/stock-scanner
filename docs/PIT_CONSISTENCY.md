# Point-in-Time Data Consistency Implementation

## Overview

This document describes the point-in-time (PIT) data consistency features implemented to eliminate look-ahead bias and ensure alignment between backtesting and online scoring.

## Problem Statement

### Look-Ahead Bias

Look-ahead bias occurs when analysis uses information that wasn't available at the time of decision-making:

- Using future price data for current decisions
- Accessing financial reports before publication
- Computing indicators with data from after the analysis time

### Time Misalignment

Different parts of the system may use inconsistent time cutoffs:

- Backtest using data up to time T1
- Online scoring using data up to time T2
- Results are not comparable if T1 ≠ T2

## Solution Architecture

### Core Components

#### 1. Timeline Correction (`as_of` Parameter)

All data retrieval functions now accept an `as_of` parameter that enforces a strict time cutoff:

```python
# Example: Get stock data as of specific time
df = await provider.get_stock_data(
    stock_code="600000",
    market_type="A",
    start_date="20230101",
    end_date="20230630",
    as_of=datetime(2023, 6, 30, 15, 0, 0)  # Only data before this time
)
```

**Implementation locations:**
- `services/stock_data_provider.py`: `get_stock_data()`, `get_multiple_stocks_data()`
- `services/factor_scoring.py`: `score()`
- `services/quant/backtest.py`: `run_backtest()`

#### 2. Snapshot Versioning

Factor snapshots now track temporal metadata:

| Field | Description |
|-------|-------------|
| `ts` | When the snapshot was created |
| `as_of` | When the snapshot becomes valid (point-in-time) |
| `data_as_of` | Latest timestamp of underlying data |

**Database schema** (`utils/persistence.py`):

```sql
CREATE TABLE IF NOT EXISTS factor_snapshots (
    ts TIMESTAMP,
    request_id VARCHAR,
    symbol VARCHAR,
    factor_id VARCHAR,
    factor_name VARCHAR,
    value DOUBLE,
    z DOUBLE,
    weight DOUBLE,
    contrib DOUBLE,
    window INTEGER,
    market_type VARCHAR,
    as_of TIMESTAMP,        -- NEW: snapshot validity time
    data_as_of TIMESTAMP    -- NEW: latest data timestamp
)
```

#### 3. Validation Framework

`utils/pit_validation.py` provides utilities for enforcing consistency:

```python
from utils.pit_validation import PITValidator

# Validate no future data
PITValidator.assert_no_future_data(df, as_of=datetime(2023, 6, 30))

# Generate quality report
report = PITValidator.generate_data_quality_report(df, as_of, symbol="600000")

# Compare backtest vs online data
comparison = PITValidator.compare_backtest_online_data(
    backtest_data=bt_df,
    online_data=ol_df,
    as_of=datetime(2023, 6, 30)
)
```

## Usage Patterns

### Pattern 1: Historical Analysis with PIT

```python
from datetime import datetime
from services.factor_scoring import FactorScoringEngine, FactorDef

# Analyze as if it were June 30, 2023
as_of = datetime(2023, 6, 30, 15, 0, 0)

engine = FactorScoringEngine()
result = await engine.score(
    symbols=["600000", "000001"],
    factors=[FactorDef(id="momentum", weight=1.0)],
    market_type="A",
    as_of=as_of,  # Only uses data available by this time
)
```

### Pattern 2: Consistent Backtest vs Online Scoring

```python
from services.quant.backtest import run_backtest, BacktestParams

as_of = datetime(2023, 6, 30)

# Backtest
bt_result = run_backtest(
    symbols=["600000"],
    market_type="A",
    params=BacktestParams(start_date="20230101", end_date="20230630"),
    as_of=as_of,
)

# Online scoring (same as_of)
score_result = await engine.score(
    symbols=["600000"],
    factors=[...],
    as_of=as_of,
)

# Both use identical data cutoff → results are comparable
```

### Pattern 3: Data Quality Validation

```python
from utils.pit_validation import PITValidator

# Fetch data
df = await provider.get_stock_data("600000", as_of=as_of)

# Validate
try:
    PITValidator.assert_no_future_data(df, as_of, context="Stock 600000")
    print("✓ No look-ahead bias detected")
except AssertionError as e:
    print(f"✗ Look-ahead bias: {e}")

# Generate report
report = PITValidator.generate_data_quality_report(df, as_of, "600000")
print(f"Valid records: {report['valid_records']}/{report['total_records']}")
```

## Testing

### Unit Tests

`tests/test_pit_consistency.py` provides comprehensive test coverage:

```bash
pytest tests/test_pit_consistency.py -v
```

**Test categories:**
1. Data truncation by `as_of`
2. Snapshot storage with temporal fields
3. Factor scoring respects `as_of`
4. Backtest and online consistency
5. Look-ahead bias detection
6. Data quality reporting

### Integration Example

Run the complete demonstration:

```bash
python examples/pit_consistency_example.py
```

This demonstrates:
- Online scoring with PIT constraint
- Backtest with same constraint
- Data quality validation
- Look-ahead bias detection
- Announcement delay validation

## Data Flow

### Before (Without PIT)

```
Data Source → API → Analysis
                     ↓
                  Uses all available data (may include future)
```

### After (With PIT)

```
Data Source → API → as_of Filter → Analysis
                         ↓
                    Only data where timestamp ≤ as_of
                         ↓
                    Metadata recorded in snapshots
```

## Validation Checklist

When implementing new features, ensure:

- [ ] All data fetches accept optional `as_of` parameter
- [ ] Data is truncated to `as_of` before analysis
- [ ] Snapshots record `as_of` and `data_as_of` timestamps
- [ ] Tests verify no future data leakage
- [ ] Quality reports generated for suspicious data
- [ ] Backtest and online paths use identical data access

## Performance Considerations

### Caching

The `as_of` parameter is included in cache keys, ensuring:
- Different `as_of` times don't share cache
- Historical queries can be cached independently
- Cache invalidation respects temporal boundaries

### Database Indexing

For optimal query performance on DuckDB:

```sql
-- Recommended index for as_of queries
CREATE INDEX idx_as_of ON factor_snapshots(as_of, symbol, factor_id);
```

## Common Pitfalls

### ❌ Wrong: Mixing as_of times

```python
# Backtest uses one time
bt = run_backtest(..., as_of=datetime(2023, 6, 30))

# Online scoring uses different time
score = await engine.score(..., as_of=datetime(2023, 7, 15))

# Results are NOT comparable!
```

### ✓ Correct: Consistent as_of

```python
as_of = datetime(2023, 6, 30)

bt = run_backtest(..., as_of=as_of)
score = await engine.score(..., as_of=as_of)

# Results are comparable
```

### ❌ Wrong: Ignoring validation errors

```python
df = get_data(...)
# No validation - may contain future data
result = compute_factors(df)
```

### ✓ Correct: Always validate

```python
df = get_data(..., as_of=as_of)
PITValidator.assert_no_future_data(df, as_of)  # Raises on violation
result = compute_factors(df)
```

## Migration Guide

### For Existing Code

1. **Add `as_of` parameters**:
   ```python
   # Before
   result = await engine.score(symbols, factors)
   
   # After
   result = await engine.score(symbols, factors, as_of=datetime.utcnow())
   ```

2. **Update database schema**:
   ```python
   # Existing snapshots won't have as_of/data_as_of
   # New code handles gracefully, but consider migration:
   # - Add columns with default value = ts
   # - Backfill from existing ts column
   ```

3. **Add validation**:
   ```python
   # Before
   df = await provider.get_stock_data(code)
   
   # After
   df = await provider.get_stock_data(code, as_of=as_of)
   PITValidator.assert_no_future_data(df, as_of)
   ```

## Future Enhancements

1. **Corporate Actions Timeline**:
   - Track when dividends, splits announced vs effective
   - Adjust prices only after announcement

2. **Financial Statement Versioning**:
   - Store restatements with revision timestamps
   - Query by report date AND as_of date

3. **Real-time Streaming**:
   - Apply PIT constraints to streaming data
   - Buffer data until officially published

4. **Automated Quality Monitoring**:
   - Scheduled jobs to scan for time violations
   - Alerts on detected look-ahead bias
   - Dashboard showing validation metrics

## References

- **Code**: `utils/persistence.py`, `services/stock_data_provider.py`, `services/factor_scoring.py`
- **Tests**: `tests/test_pit_consistency.py`
- **Examples**: `examples/pit_consistency_example.py`
- **Validation**: `utils/pit_validation.py`

## Support

For questions or issues:
1. Check test examples in `tests/test_pit_consistency.py`
2. Run demonstration: `python examples/pit_consistency_example.py`
3. Review validation utilities in `utils/pit_validation.py`
