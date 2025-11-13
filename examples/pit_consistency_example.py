"""
Example demonstrating point-in-time data consistency for backtesting and online scoring.

This example shows how to:
1. Use as_of parameter to prevent look-ahead bias
2. Ensure backtest and online scoring use consistent data
3. Validate data quality and time alignment
"""
import asyncio
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.factor_scoring import FactorScoringEngine, FactorDef
from services.quant.backtest import run_backtest, BacktestParams
from services.stock_data_provider import StockDataProvider
from utils.pit_validation import PITValidator
from utils.logger import get_logger

logger = get_logger()


async def demonstrate_pit_consistency():
    """
    Demonstrate point-in-time consistency between backtest and online scoring.
    """
    logger.info("=" * 80)
    logger.info("Point-in-Time Data Consistency Demonstration")
    logger.info("=" * 80)
    
    # Define test parameters
    symbols = ["600000", "000001", "000002"]
    market_type = "A"
    
    # Critical: use same as_of time for both backtest and online scoring
    as_of = datetime(2023, 6, 30, 15, 0, 0)  # End of trading day
    
    logger.info(f"\nTesting with as_of={as_of.isoformat()}")
    logger.info(f"Symbols: {symbols}")
    
    # Step 1: Online scoring with point-in-time constraint
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Online Factor Scoring (with PIT constraint)")
    logger.info("=" * 80)
    
    engine = FactorScoringEngine()
    factors = [
        FactorDef(id="momentum", name="20-day Momentum", weight=1.0, params={"window": 20}),
        FactorDef(id="ma_dev", name="MA Deviation", weight=0.8, params={"period": 20}),
        FactorDef(id="vol", name="Volatility", weight=-0.5, params={"window": 20}),
    ]
    
    try:
        scoring_result = await engine.score(
            symbols=symbols,
            factors=factors,
            market_type=market_type,
            window=20,
            start_date="20230101",
            end_date="20230630",
            as_of=as_of,  # Point-in-time constraint
            page_size=10,
        )
        
        logger.info(f"Scoring completed: {scoring_result.get('total', 0)} results")
        
        # Display top results
        for i, result in enumerate(scoring_result.get("results", [])[:3], 1):
            logger.info(
                f"  {i}. {result['symbol']}: "
                f"score={result['total_score']:.3f}, "
                f"risk={result.get('risk_tag', 'N/A')}"
            )
    
    except Exception as e:
        logger.warning(f"Scoring skipped (data unavailable): {e}")
        scoring_result = None
    
    # Step 2: Backtest with same point-in-time constraint
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Backtest (with same PIT constraint)")
    logger.info("=" * 80)
    
    params = BacktestParams(
        start_date="20230101",
        end_date="20230630",
        initial_cash=100000.0,
    )
    
    try:
        backtest_result = run_backtest(
            symbols=symbols,
            market_type=market_type,
            params=params,
            as_of=as_of,  # Same point-in-time constraint
        )
        
        logger.info(f"Backtest completed: {backtest_result['start']} to {backtest_result['end']}")
        logger.info(f"  Total Return: {backtest_result['metrics']['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {backtest_result['metrics']['sharpe']:.2f}")
        logger.info(f"  Max Drawdown: {backtest_result['metrics']['max_drawdown']:.2%}")
    
    except Exception as e:
        logger.warning(f"Backtest skipped (data unavailable): {e}")
        backtest_result = None
    
    # Step 3: Validate data consistency
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Data Quality Validation")
    logger.info("=" * 80)
    
    provider = StockDataProvider()
    
    for symbol in symbols[:2]:  # Validate first 2 symbols
        try:
            # Fetch data with as_of constraint
            df = await provider.get_stock_data(
                stock_code=symbol,
                market_type=market_type,
                start_date="20230101",
                end_date="20230630",
                as_of=as_of,
            )
            
            if not df.empty:
                # Generate quality report
                report = PITValidator.generate_data_quality_report(
                    df=df,
                    as_of=as_of,
                    symbol=symbol
                )
                
                logger.info(f"\n  Symbol: {symbol}")
                logger.info(f"    Records: {report['total_records']}")
                logger.info(f"    Valid: {report['valid_records']}")
                logger.info(f"    Future data: {report['has_future_data']}")
                
                if report['issues']:
                    logger.warning(f"    Issues: {', '.join(report['issues'])}")
                else:
                    logger.info(f"    Status: ✓ All checks passed")
                
                # Assert no future data
                try:
                    PITValidator.assert_no_future_data(df, as_of, context=symbol)
                    logger.info(f"    PIT Assertion: ✓ PASSED")
                except AssertionError as e:
                    logger.error(f"    PIT Assertion: ✗ FAILED - {e}")
        
        except Exception as e:
            logger.warning(f"  Symbol {symbol}: Validation skipped - {e}")
    
    # Step 4: Demonstrate look-ahead bias detection
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Look-Ahead Bias Detection")
    logger.info("=" * 80)
    
    import pandas as pd
    import numpy as np
    
    # Create mock data with intentional look-ahead bias
    past_dates = pd.date_range(
        start=as_of - timedelta(days=30),
        end=as_of,
        freq="D"
    )
    future_dates = pd.date_range(
        start=as_of + timedelta(days=1),
        end=as_of + timedelta(days=10),
        freq="D"
    )
    all_dates = past_dates.append(future_dates)
    
    biased_data = pd.DataFrame({
        "Close": np.random.rand(len(all_dates)) * 100,
    }, index=all_dates)
    
    logger.info(f"  Created test data: {len(past_dates)} past + {len(future_dates)} future dates")
    
    # This should fail validation
    is_valid, error_msg = PITValidator.validate_data_timestamp(biased_data, as_of)
    
    if not is_valid:
        logger.info(f"  ✓ Look-ahead bias correctly detected!")
        logger.info(f"    Error: {error_msg}")
    else:
        logger.error(f"  ✗ Look-ahead bias NOT detected (validation bug)")
    
    # Clean data (no future dates)
    clean_data = biased_data[biased_data.index <= as_of]
    is_valid, error_msg = PITValidator.validate_data_timestamp(clean_data, as_of)
    
    if is_valid:
        logger.info(f"  ✓ Clean data passed validation")
    else:
        logger.error(f"  ✗ Clean data failed validation: {error_msg}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info("""
    Point-in-Time Consistency Features Demonstrated:
    
    1. ✓ as_of parameter prevents future data leakage
    2. ✓ Backtest and online scoring use same data cutoff
    3. ✓ Snapshot store records as_of and data_as_of timestamps
    4. ✓ Data quality validation detects time misalignment
    5. ✓ Assertions catch look-ahead bias
    
    Key Principles:
    - Always specify as_of for historical analysis
    - Use same as_of for backtest and online scoring comparison
    - Validate data timestamps before computation
    - Record metadata (as_of, data_as_of) in snapshots
    - Run quality checks regularly
    """)
    
    logger.info("\nDemonstration complete!")


async def demonstrate_announcement_delay():
    """
    Demonstrate proper handling of announcement data with publication delays.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Announcement Data Timeline Validation")
    logger.info("=" * 80)
    
    # Scenario 1: After-hours announcement
    announcement_time = datetime(2023, 6, 15, 16, 30)  # 4:30 PM
    available_time = datetime(2023, 6, 16, 9, 30)  # Next day 9:30 AM
    
    is_valid, msg = PITValidator.check_announcement_delay(
        announcement_time=announcement_time,
        available_time=available_time,
        min_delay_hours=12.0
    )
    
    logger.info(f"\nScenario 1: After-hours announcement")
    logger.info(f"  Announced: {announcement_time}")
    logger.info(f"  Available: {available_time}")
    logger.info(f"  Status: {'✓ VALID' if is_valid else '✗ INVALID'}")
    logger.info(f"  {msg}")
    
    # Scenario 2: Invalid - immediate availability
    announcement_time2 = datetime(2023, 6, 15, 10, 0)
    available_time2 = datetime(2023, 6, 15, 10, 5)  # 5 minutes later
    
    is_valid2, msg2 = PITValidator.check_announcement_delay(
        announcement_time=announcement_time2,
        available_time=available_time2,
        min_delay_hours=12.0
    )
    
    logger.info(f"\nScenario 2: Immediate availability (INVALID)")
    logger.info(f"  Announced: {announcement_time2}")
    logger.info(f"  Available: {available_time2}")
    logger.info(f"  Status: {'✓ VALID' if is_valid2 else '✗ INVALID (Expected)'}")
    logger.info(f"  {msg2}")


def main():
    """Run all demonstrations"""
    asyncio.run(demonstrate_pit_consistency())
    asyncio.run(demonstrate_announcement_delay())


if __name__ == "__main__":
    main()
