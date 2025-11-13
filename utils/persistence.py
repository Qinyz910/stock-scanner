import os
import uuid
from datetime import datetime
from typing import List, Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger()


class SnapshotStore:
    """
    Persist factor snapshots optionally to DuckDB if available, else CSV files.
    """

    def __init__(self, base_dir: str = "data", duckdb_path: Optional[str] = None):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.duckdb_path = duckdb_path or os.path.join(self.base_dir, "factors.duckdb")
        self._duckdb = None

        try:
            import duckdb  # type: ignore

            self._duckdb = duckdb.connect(self.duckdb_path)
            # Create table upfront with as_of and data_as_of for point-in-time consistency
            self._duckdb.execute(
                """
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
                    as_of TIMESTAMP,
                    data_as_of TIMESTAMP
                )
                """
            )
            logger.info("DuckDB snapshot store enabled at %s", self.duckdb_path)
        except Exception as e:
            logger.warning("DuckDB unavailable, using CSV snapshot store: %s", str(e))
            self._duckdb = None

    def save(self, df: pd.DataFrame) -> None:
        """
        Save snapshot records.
        df must include columns: ts, request_id, symbol, factor_id, factor_name, value, z, weight, contrib, window, market_type, as_of, data_as_of
        """
        if df is None or df.empty:
            return

        if self._duckdb is not None:
            try:
                self._duckdb.register("tmp_df", df)
                self._duckdb.execute(
                    """
                    INSERT INTO factor_snapshots
                    SELECT ts, request_id, symbol, factor_id, factor_name, value, z, weight, contrib, window, market_type, as_of, data_as_of
                    FROM tmp_df
                    """
                )
                self._duckdb.unregister("tmp_df")
                return
            except Exception as e:
                logger.error("Failed to insert into DuckDB, fallback to CSV: %s", str(e))

        # CSV fallback
        try:
            snap_dir = os.path.join(self.base_dir, "factor_snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(snap_dir, f"snap_{ts}_{uuid.uuid4().hex[:8]}.csv")
            df.to_csv(file_path, index=False)
            logger.info("Saved factor snapshot CSV at %s", file_path)
        except Exception as e:
            logger.error("Failed to save snapshot CSV: %s", str(e))

    def query(self, as_of: datetime, symbols: Optional[List[str]] = None, factor_ids: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Query snapshots as of a given time.
        Returns the most recent snapshot for each symbol/factor where as_of <= specified time.
        """
        if self._duckdb is None:
            logger.warning("DuckDB not available, cannot query snapshots")
            return None

        try:
            where_clauses = [f"as_of <= '{as_of.isoformat()}'"]
            if symbols:
                symbol_list = "', '".join(symbols)
                where_clauses.append(f"symbol IN ('{symbol_list}')")
            if factor_ids:
                factor_list = "', '".join(factor_ids)
                where_clauses.append(f"factor_id IN ('{factor_list}')")
            
            where_clause = " AND ".join(where_clauses)
            
            query = f"""
                SELECT * FROM factor_snapshots
                WHERE {where_clause}
                ORDER BY as_of DESC, ts DESC
            """
            
            result = self._duckdb.execute(query).fetchdf()
            return result
        except Exception as e:
            logger.error("Failed to query snapshots: %s", str(e))
            return None
