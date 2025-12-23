"""
Backtest repository for storing runs, results, and signals.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from atm.config import DatabaseConfig
from atm.repo.database_repo import DatabaseRepo


class BacktestRepo:
    """
    Repository to persist backtest runs, per-stock results, and signals.

    Tables:
    - backtest_runs: one record per backtest command (params stored as JSONB)
    - backtest_results: per-stock aggregate backtest metrics, linked to backtest_runs
    - backtest_signals: per-signal records, linked to backtest_runs and ts_code
    """

    def __init__(self, db_config: DatabaseConfig, schema: Optional[str] = None):
        self.db_config = db_config
        self.schema = schema or db_config.schema
        self.run_table = "backtest_runs"
        self.result_table = "backtest_results"
        self.signal_table = "backtest_signals"
        self._engine: Optional[Engine] = None

    def _get_engine(self) -> Engine:
        if self._engine is None:
            repo = DatabaseRepo(self.db_config, table_name=self.run_table, schema=self.schema)
            self._engine = repo._get_engine()  # reuse engine creation
        return self._engine

    @staticmethod
    def _generate_run_id() -> str:
        """
        Generate a short hash-like run id (8 chars).

        Uses urlsafe base64 of 6 random bytes -> 8 characters, no padding.
        """
        import base64
        import secrets

        return base64.urlsafe_b64encode(secrets.token_bytes(6)).decode("ascii").rstrip("=")

    def ensure_tables(self) -> None:
        """Create tables if not exist."""
        engine = self._get_engine()
        ddl = f"""
        CREATE SCHEMA IF NOT EXISTS "{self.schema}";

        CREATE TABLE IF NOT EXISTS "{self.schema}"."{self.run_table}" (
            id              VARCHAR(16) PRIMARY KEY,
            strategy_name   TEXT NOT NULL,
            command         TEXT,
            params          JSONB,
            kline_type      TEXT,
            start_date      DATE,
            end_date        DATE,
            start_time      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            end_time        TIMESTAMPTZ,
            status          TEXT DEFAULT 'running', -- running / success / failed
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS "{self.schema}"."{self.result_table}" (
            id              BIGSERIAL PRIMARY KEY,
            run_id          VARCHAR(16) NOT NULL REFERENCES "{self.schema}"."{self.run_table}"(id) ON DELETE CASCADE,
            ts_code         TEXT NOT NULL,
            metrics         JSONB,
            stats           JSONB,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_{self.result_table}_run ON "{self.schema}"."{self.result_table}"(run_id);
        CREATE INDEX IF NOT EXISTS idx_{self.result_table}_ts ON "{self.schema}"."{self.result_table}"(ts_code);

        CREATE TABLE IF NOT EXISTS "{self.schema}"."{self.signal_table}" (
            id              BIGSERIAL PRIMARY KEY,
            run_id          VARCHAR(16) NOT NULL REFERENCES "{self.schema}"."{self.run_table}"(id) ON DELETE CASCADE,
            ts_code         TEXT NOT NULL,
            signal_time     TIMESTAMPTZ NOT NULL,
            signal_type     TEXT NOT NULL, -- buy / sell / add / exit / stop
            price           NUMERIC,
            size            NUMERIC,
            extra           JSONB,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_{self.signal_table}_run ON "{self.schema}"."{self.signal_table}"(run_id);
        CREATE INDEX IF NOT EXISTS idx_{self.signal_table}_ts ON "{self.schema}"."{self.signal_table}"(ts_code);
        CREATE INDEX IF NOT EXISTS idx_{self.signal_table}_time ON "{self.schema}"."{self.signal_table}"(signal_time);
        """
        with engine.begin() as conn:
            conn.execute(text(ddl))

    def insert_run(
        self,
        strategy_name: str,
        command: Optional[str],
        params: Dict[str, Any],
        kline_type: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        run_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        status: str = "running",
    ) -> str:
        """Insert a backtest run and return run_id (short hash, 8 chars)."""
        rid = run_id or self._generate_run_id()
        engine = self._get_engine()
        sql = text(
            f"""
            INSERT INTO "{self.schema}"."{self.run_table}"
            (id, strategy_name, command, params, kline_type, start_date, end_date, start_time, status)
            VALUES (:id, :strategy_name, :command, :params, :kline_type, :start_date, :end_date, :start_time, :status)
            RETURNING id;
            """
        )
        with engine.begin() as conn:
            result = conn.execute(
                sql,
                {
                    "id": rid,
                    "strategy_name": strategy_name,
                    "command": command,
                    "params": json.dumps(params),
                    "kline_type": kline_type,
                    "start_date": start_date,
                    "end_date": end_date,
                    "start_time": start_time or datetime.utcnow(),
                    "status": status,
                },
            )
            run_id_db = result.scalar_one()
        return str(run_id_db)

    def update_run_status(
        self,
        run_id: str,
        status: str,
        end_time: Optional[datetime] = None,
    ) -> None:
        """Update run status and end_time."""
        engine = self._get_engine()
        sql = text(
            f"""
            UPDATE "{self.schema}"."{self.run_table}"
            SET status = :status,
                end_time = :end_time
            WHERE id = :run_id;
            """
        )
        with engine.begin() as conn:
            conn.execute(
                sql,
                {
                    "status": status,
                    "end_time": end_time or datetime.utcnow(),
                    "run_id": run_id,
                },
            )

    def insert_results(
        self, run_id: str, results: List[Dict[str, Any]]
    ) -> int:
        """
        Insert multiple per-stock results.
        Each item should contain: ts_code, metrics (dict), stats (dict).
        """
        if not results:
            return 0
        engine = self._get_engine()
        rows = [
            {
                "run_id": run_id,
                "ts_code": r.get("ts_code"),
                "metrics": json.dumps(r.get("metrics", {})),
                "stats": json.dumps(r.get("stats", {})),
            }
            for r in results
        ]
        sql = text(
            f"""
            INSERT INTO "{self.schema}"."{self.result_table}"
            (run_id, ts_code, metrics, stats)
            VALUES (:run_id, :ts_code, :metrics, :stats);
            """
        )
        with engine.begin() as conn:
            conn.execute(sql, rows)
        return len(rows)

    def insert_signals(
        self, run_id: str, ts_code: str, signals: List[Dict[str, Any]]
    ) -> int:
        """
        Insert signal records.
        Each signal dict should contain: signal_time (datetime), signal_type (str),
        price (optional), size (optional), extra (dict, optional).
        """
        if not signals:
            return 0
        engine = self._get_engine()
        rows = []
        for s in signals:
            rows.append(
                {
                    "run_id": run_id,
                    "ts_code": ts_code,
                    "signal_time": s.get("signal_time"),
                    "signal_type": s.get("signal_type"),
                    "price": s.get("price"),
                    "size": s.get("size"),
                    "extra": json.dumps(s.get("extra", {})),
                }
            )
        sql = text(
            f"""
            INSERT INTO "{self.schema}"."{self.signal_table}"
            (run_id, ts_code, signal_time, signal_type, price, size, extra)
            VALUES (:run_id, :ts_code, :signal_time, :signal_type, :price, :size, :extra);
            """
        )
        with engine.begin() as conn:
            conn.execute(sql, rows)
        return len(rows)


