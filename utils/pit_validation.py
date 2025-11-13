"""
Point-in-time data validation utilities to prevent look-ahead bias.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger()


class PITValidator:
    """
    Point-in-time validator to ensure data consistency and prevent look-ahead bias.
    """
    
    @staticmethod
    def validate_data_timestamp(
        df: pd.DataFrame, 
        as_of: datetime,
        timestamp_col: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Validate that DataFrame contains no data after as_of timestamp.
        
        Args:
            df: DataFrame with DatetimeIndex or timestamp column
            as_of: Cut-off timestamp
            timestamp_col: Optional column name if not using index
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if df.empty:
                return True, ""
            
            if timestamp_col:
                if timestamp_col not in df.columns:
                    return False, f"Timestamp column '{timestamp_col}' not found"
                timestamps = pd.to_datetime(df[timestamp_col])
            else:
                if not isinstance(df.index, pd.DatetimeIndex):
                    return False, "DataFrame must have DatetimeIndex or specify timestamp_col"
                timestamps = df.index
            
            max_timestamp = timestamps.max()
            as_of_ts = pd.Timestamp(as_of)
            
            if max_timestamp > as_of_ts:
                future_count = (timestamps > as_of_ts).sum()
                return False, (
                    f"Look-ahead bias detected: {future_count} records after as_of={as_of}. "
                    f"Latest timestamp: {max_timestamp}"
                )
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def assert_no_future_data(
        df: pd.DataFrame,
        as_of: datetime,
        context: str = ""
    ) -> None:
        """
        Assert that DataFrame contains no future data. Raises AssertionError if violated.
        
        Args:
            df: DataFrame to validate
            as_of: Cut-off timestamp
            context: Context information for error message
        """
        is_valid, error_msg = PITValidator.validate_data_timestamp(df, as_of)
        if not is_valid:
            full_msg = f"[{context}] {error_msg}" if context else error_msg
            logger.error(full_msg)
            raise AssertionError(full_msg)
        else:
            logger.debug(f"[{context}] PIT validation passed for as_of={as_of}")
    
    @staticmethod
    def check_announcement_delay(
        announcement_time: datetime,
        available_time: datetime,
        min_delay_hours: float = 12.0
    ) -> Tuple[bool, str]:
        """
        Verify that announcement data has appropriate delay before becoming available.
        
        Args:
            announcement_time: When announcement was made
            available_time: When data becomes available for use
            min_delay_hours: Minimum required delay in hours
            
        Returns:
            Tuple of (is_valid, message)
        """
        delay = available_time - announcement_time
        delay_hours = delay.total_seconds() / 3600
        
        if delay_hours < 0:
            return False, f"Invalid: available_time before announcement_time"
        
        if delay_hours < min_delay_hours:
            return False, (
                f"Insufficient delay: {delay_hours:.1f}h < {min_delay_hours}h required. "
                f"Announcement at {announcement_time}, available at {available_time}"
            )
        
        return True, f"Valid delay: {delay_hours:.1f}h"
    
    @staticmethod
    def generate_data_quality_report(
        df: pd.DataFrame,
        as_of: datetime,
        symbol: str = "UNKNOWN"
    ) -> Dict:
        """
        Generate data quality report for given DataFrame.
        
        Returns:
            Dictionary with quality metrics
        """
        report = {
            "symbol": symbol,
            "as_of": as_of.isoformat(),
            "total_records": len(df),
            "date_range": None,
            "has_future_data": False,
            "future_data_count": 0,
            "has_gaps": False,
            "gap_count": 0,
            "valid_records": 0,
            "issues": [],
        }
        
        if df.empty:
            report["issues"].append("Empty DataFrame")
            return report
        
        try:
            # Check date range
            if isinstance(df.index, pd.DatetimeIndex):
                report["date_range"] = (
                    df.index.min().isoformat(),
                    df.index.max().isoformat()
                )
                
                # Check for future data
                as_of_ts = pd.Timestamp(as_of)
                future_mask = df.index > as_of_ts
                future_count = future_mask.sum()
                
                if future_count > 0:
                    report["has_future_data"] = True
                    report["future_data_count"] = int(future_count)
                    report["issues"].append(
                        f"Look-ahead bias: {future_count} records after as_of"
                    )
                
                report["valid_records"] = len(df) - future_count
                
                # Check for gaps (missing business days)
                if len(df) > 1:
                    date_diffs = df.index.to_series().diff()
                    # Gaps > 7 days might indicate missing data
                    large_gaps = date_diffs[date_diffs > pd.Timedelta(days=7)]
                    if len(large_gaps) > 0:
                        report["has_gaps"] = True
                        report["gap_count"] = len(large_gaps)
                        report["issues"].append(
                            f"{len(large_gaps)} large time gaps detected (>7 days)"
                        )
            
        except Exception as e:
            report["issues"].append(f"Report generation error: {str(e)}")
        
        return report
    
    @staticmethod
    def validate_factor_computation(
        factor_value: float,
        data_timestamps: pd.DatetimeIndex,
        as_of: datetime,
        factor_name: str = "unknown"
    ) -> Tuple[bool, str]:
        """
        Validate that a factor computation used only valid historical data.
        
        Args:
            factor_value: Computed factor value
            data_timestamps: Timestamps of data used in computation
            as_of: Point-in-time timestamp
            factor_name: Name of factor for logging
            
        Returns:
            Tuple of (is_valid, message)
        """
        if np.isnan(factor_value) or np.isinf(factor_value):
            return True, f"Factor {factor_name} is NaN/Inf (acceptable)"
        
        if data_timestamps.empty:
            return False, f"Factor {factor_name}: no data timestamps provided"
        
        max_timestamp = data_timestamps.max()
        as_of_ts = pd.Timestamp(as_of)
        
        if max_timestamp > as_of_ts:
            return False, (
                f"Factor {factor_name}: used data after as_of. "
                f"Latest data: {max_timestamp}, as_of: {as_of_ts}"
            )
        
        return True, f"Factor {factor_name}: validation passed"
    
    @staticmethod
    def compare_backtest_online_data(
        backtest_data: pd.DataFrame,
        online_data: pd.DataFrame,
        as_of: datetime,
        tolerance: float = 1e-6
    ) -> Dict:
        """
        Compare data used in backtest vs online scoring to ensure consistency.
        
        Args:
            backtest_data: Data used in backtest
            online_data: Data used in online scoring
            as_of: Point-in-time timestamp both should respect
            tolerance: Numerical tolerance for comparison
            
        Returns:
            Dictionary with comparison results
        """
        result = {
            "consistent": True,
            "as_of": as_of.isoformat(),
            "issues": [],
            "backtest_records": len(backtest_data),
            "online_records": len(online_data),
        }
        
        try:
            # Both should respect as_of
            as_of_ts = pd.Timestamp(as_of)
            
            if not backtest_data.empty:
                bt_max = backtest_data.index.max() if isinstance(backtest_data.index, pd.DatetimeIndex) else None
                if bt_max and bt_max > as_of_ts:
                    result["consistent"] = False
                    result["issues"].append(f"Backtest data exceeds as_of: {bt_max}")
            
            if not online_data.empty:
                ol_max = online_data.index.max() if isinstance(online_data.index, pd.DatetimeIndex) else None
                if ol_max and ol_max > as_of_ts:
                    result["consistent"] = False
                    result["issues"].append(f"Online data exceeds as_of: {ol_max}")
            
            # Compare common columns if both have data
            if not backtest_data.empty and not online_data.empty:
                common_cols = set(backtest_data.columns) & set(online_data.columns)
                if common_cols:
                    # Align on index
                    common_index = backtest_data.index.intersection(online_data.index)
                    if len(common_index) > 0:
                        for col in common_cols:
                            bt_vals = backtest_data.loc[common_index, col]
                            ol_vals = online_data.loc[common_index, col]
                            
                            # Check numerical consistency
                            if np.issubdtype(bt_vals.dtype, np.number):
                                diff = np.abs(bt_vals - ol_vals)
                                max_diff = diff.max()
                                if max_diff > tolerance:
                                    result["consistent"] = False
                                    result["issues"].append(
                                        f"Column '{col}' differs by up to {max_diff:.2e}"
                                    )
        
        except Exception as e:
            result["consistent"] = False
            result["issues"].append(f"Comparison error: {str(e)}")
        
        return result


def create_pit_assertion_decorator(as_of: datetime):
    """
    Create a decorator that validates function output for point-in-time consistency.
    
    Usage:
        @create_pit_assertion_decorator(as_of=datetime(2023, 6, 30))
        def compute_factors(data):
            return factors
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # If result is DataFrame, validate
            if isinstance(result, pd.DataFrame):
                PITValidator.assert_no_future_data(
                    result, 
                    as_of, 
                    context=f"{func.__name__}"
                )
            
            return result
        return wrapper
    return decorator
