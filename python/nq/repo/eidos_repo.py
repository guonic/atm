"""
Eidos (Universal Backtest Attribution System) Repository.

Repository for storing and retrieving backtest attribution data.
Supports any quantitative model (GNN, time series, linear, etc.).
"""

import json
import logging
import secrets
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from nq.config import DatabaseConfig
from nq.repo.database_repo import DatabaseRepo


def generate_exp_id() -> str:
    """
    Generate a short 8-character hexadecimal experiment ID.
    
    Returns:
        8-character hex string (e.g., 'a1b2c3d4').
    """
    return secrets.token_hex(4)


class EidosExperimentRepo:
    """Repository for backtest experiment metadata."""

    def __init__(self, db_config: DatabaseConfig, schema: str = "eidos"):
        """
        Initialize experiment repository.

        Args:
            db_config: Database configuration.
            schema: Database schema name (default: 'eidos').
        """
        self.db_config = db_config
        self.schema = schema
        self.table_name = "bt_experiment"
        self._engine: Optional[Engine] = None

    def _get_engine(self) -> Engine:
        """Get or create database engine."""
        if self._engine is None:
            repo = DatabaseRepo(
                self.db_config, table_name=self.table_name, schema=self.schema
            )
            self._engine = repo._get_engine()
        return self._engine

    def create_experiment(
        self,
        name: str,
        model_type: Optional[str] = None,
        engine_type: Optional[str] = None,
        start_date: date = None,
        end_date: date = None,
        config: Dict[str, Any] = None,
        metrics_summary: Optional[Dict[str, Any]] = None,
        version: int = 1,
        status: str = "running",
    ) -> str:
        """
        Create a new experiment and return exp_id.

        Args:
            name: Experiment name.
            model_type: Model type (e.g., 'GNN', 'GRU', 'Linear').
            engine_type: Engine type (e.g., 'Qlib', 'Backtrader').
            start_date: Start date.
            end_date: End date.
            config: Configuration parameters (JSONB).
            metrics_summary: Summary metrics (JSONB).
            version: Version number.
            status: Status ('running', 'completed', 'failed').

        Returns:
            Experiment ID (UUID as string).
        """
        engine = self._get_engine()
        sql = text(
            f"""
            INSERT INTO "{self.schema}"."{self.table_name}"
            (exp_id, name, model_type, engine_type, start_date, end_date, config, metrics_summary, version, status)
            VALUES (:exp_id, :name, :model_type, :engine_type, :start_date, :end_date, :config, :metrics_summary, :version, :status);
            """
        )
        # Generate 8-character hex ID
        exp_id = generate_exp_id()
        
        with engine.begin() as conn:
            conn.execute(
                sql,
                {
                    "exp_id": exp_id,
                    "name": name,
                    "model_type": model_type,
                    "engine_type": engine_type,
                    "start_date": start_date,
                    "end_date": end_date,
                    "config": json.dumps(config or {}),
                    "metrics_summary": json.dumps(metrics_summary) if metrics_summary else None,
                    "version": version,
                    "status": status,
                },
            )
        return exp_id

    def list_experiments(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List experiments.

        Args:
            limit: Maximum number of experiments to return.
            offset: Offset for pagination.
            status: Filter by status (optional).

        Returns:
            List of experiment dictionaries.
        """
        engine = self._get_engine()
        sql = text(
            f"""
            SELECT exp_id, name, model_type, engine_type, start_date, end_date,
                   config, metrics_summary, version, status, created_at, updated_at
            FROM "{self.schema}"."{self.table_name}"
            {"WHERE status = :status" if status else ""}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset;
            """
        )
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        with engine.connect() as conn:
            result = conn.execute(sql, params)
            rows = result.fetchall()

        experiments = []
        for row in rows:
            # Handle JSONB columns - they may already be dicts or strings
            config = row[6]
            if config is not None:
                if isinstance(config, str):
                    config = json.loads(config)
                elif not isinstance(config, dict):
                    config = {}
            else:
                config = {}
            
            metrics_summary = row[7]
            if metrics_summary is not None:
                if isinstance(metrics_summary, str):
                    metrics_summary = json.loads(metrics_summary)
                elif not isinstance(metrics_summary, dict):
                    metrics_summary = {}
            else:
                metrics_summary = {}
            
            exp = {
                "exp_id": str(row[0]),
                "name": row[1],
                "model_type": row[2],
                "engine_type": row[3],
                "start_date": row[4].isoformat() if row[4] else None,
                "end_date": row[5].isoformat() if row[5] else None,
                "config": config,
                "metrics_summary": metrics_summary,
                "version": row[8],
                "status": row[9],
                "created_at": row[10].isoformat() if row[10] else None,
                "updated_at": row[11].isoformat() if row[11] else None,
            }
            experiments.append(exp)

        return experiments

    def update_experiment(
        self,
        exp_id: str,
        status: Optional[str] = None,
        metrics_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update experiment status and/or metrics.

        Args:
            exp_id: Experiment ID.
            status: New status.
            metrics_summary: Updated metrics summary.
        """
        engine = self._get_engine()
        updates = []
        params = {"exp_id": exp_id}

        if status:
            updates.append("status = :status")
            params["status"] = status

        if metrics_summary:
            updates.append("metrics_summary = :metrics_summary")
            params["metrics_summary"] = json.dumps(metrics_summary)

        if not updates:
            return

        sql = text(
            f"""
            UPDATE "{self.schema}"."{self.table_name}"
            SET {', '.join(updates)}
            WHERE exp_id = :exp_id;
            """
        )
        with engine.begin() as conn:
            conn.execute(sql, params)

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment by ID.

        Args:
            exp_id: Experiment ID.

        Returns:
            Experiment data or None if not found.
        """
        engine = self._get_engine()
        sql = text(
            f"""
            SELECT exp_id, name, model_type, engine_type, start_date, end_date,
                   config, metrics_summary, version, status, created_at, updated_at
            FROM "{self.schema}"."{self.table_name}"
            WHERE exp_id = :exp_id;
            """
        )
        with engine.connect() as conn:
            result = conn.execute(sql, {"exp_id": exp_id})
            row = result.fetchone()
            if row:
                exp = dict(row._mapping)
                # Handle JSONB columns - they may already be dicts or strings
                if "config" in exp and exp["config"] is not None:
                    if isinstance(exp["config"], str):
                        exp["config"] = json.loads(exp["config"])
                    elif not isinstance(exp["config"], dict):
                        exp["config"] = {}
                else:
                    exp["config"] = {}
                
                if "metrics_summary" in exp and exp["metrics_summary"] is not None:
                    if isinstance(exp["metrics_summary"], str):
                        exp["metrics_summary"] = json.loads(exp["metrics_summary"])
                    elif not isinstance(exp["metrics_summary"], dict):
                        exp["metrics_summary"] = {}
                else:
                    exp["metrics_summary"] = {}
                
                # Convert UUID and dates to strings
                if "exp_id" in exp:
                    exp["exp_id"] = str(exp["exp_id"])
                if "start_date" in exp and exp["start_date"]:
                    exp["start_date"] = exp["start_date"].isoformat()
                if "end_date" in exp and exp["end_date"]:
                    exp["end_date"] = exp["end_date"].isoformat()
                if "created_at" in exp and exp["created_at"]:
                    exp["created_at"] = exp["created_at"].isoformat()
                if "updated_at" in exp and exp["updated_at"]:
                    exp["updated_at"] = exp["updated_at"].isoformat()
                
                return exp
        return None


class EidosLedgerRepo:
    """Repository for account ledger (daily NAV)."""

    def __init__(self, db_config: DatabaseConfig, schema: str = "eidos"):
        """
        Initialize ledger repository.

        Args:
            db_config: Database configuration.
            schema: Database schema name (default: 'eidos').
        """
        self.db_config = db_config
        self.schema = schema
        self.table_name = "bt_ledger"
        self._engine: Optional[Engine] = None

    def _get_engine(self) -> Engine:
        """Get or create database engine."""
        if self._engine is None:
            repo = DatabaseRepo(
                self.db_config, table_name=self.table_name, schema=self.schema
            )
            self._engine = repo._get_engine()
        return self._engine

    def batch_upsert(
        self,
        exp_id: str,
        ledger_data: List[Dict[str, Any]],
        batch_size: int = 1000,
    ) -> int:
        """
        Batch upsert ledger data.

        Args:
            exp_id: Experiment ID.
            ledger_data: List of ledger records. Each record should contain:
                - date: DATE
                - nav: NUMERIC
                - cash: NUMERIC (optional)
                - market_value: NUMERIC (optional)
                - deal_amount: NUMERIC (optional)
                - turnover_rate: FLOAT (optional)
                - pos_count: INTEGER (optional)
            batch_size: Batch size for insertion.

        Returns:
            Number of records inserted/updated.
        """
        if not ledger_data:
            return 0

        engine = self._get_engine()
        total_inserted = 0

        for i in range(0, len(ledger_data), batch_size):
            batch = ledger_data[i : i + batch_size]
            sql = text(
                f"""
                INSERT INTO "{self.schema}"."{self.table_name}"
                (exp_id, date, nav, cash, market_value, deal_amount, turnover_rate, pos_count)
                VALUES (:exp_id, :date, :nav, :cash, :market_value, :deal_amount, :turnover_rate, :pos_count)
                ON CONFLICT (exp_id, date) DO UPDATE SET
                    nav = EXCLUDED.nav,
                    cash = EXCLUDED.cash,
                    market_value = EXCLUDED.market_value,
                    deal_amount = EXCLUDED.deal_amount,
                    turnover_rate = EXCLUDED.turnover_rate,
                    pos_count = EXCLUDED.pos_count;
                """
            )
            rows = [
                {
                    "exp_id": exp_id,
                    "date": record["date"],
                    "nav": record.get("nav"),
                    "cash": record.get("cash"),
                    "market_value": record.get("market_value"),
                    "deal_amount": record.get("deal_amount"),
                    "turnover_rate": record.get("turnover_rate"),
                    "pos_count": record.get("pos_count"),
                }
                for record in batch
            ]
            with engine.begin() as conn:
                result = conn.execute(sql, rows)
                total_inserted += result.rowcount or len(rows)

        return total_inserted

    def get_ledger(
        self,
        exp_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get ledger data for an experiment.

        Args:
            exp_id: Experiment ID.
            start_date: Start date (optional).
            end_date: End date (optional).

        Returns:
            List of ledger records.
        """
        engine = self._get_engine()
        sql = f"""
            SELECT exp_id, date, nav, cash, market_value, deal_amount, turnover_rate, pos_count, created_at
            FROM "{self.schema}"."{self.table_name}"
            WHERE exp_id = :exp_id
        """
        params = {"exp_id": exp_id}

        if start_date:
            sql += " AND date >= :start_date"
            params["start_date"] = start_date
        if end_date:
            sql += " AND date <= :end_date"
            params["end_date"] = end_date

        sql += " ORDER BY date ASC;"

        with engine.connect() as conn:
            result = conn.execute(text(sql), params)
            return [dict(row._mapping) for row in result]


class EidosTradesRepo:
    """Repository for trade orders."""

    def __init__(self, db_config: DatabaseConfig, schema: str = "eidos"):
        """
        Initialize trades repository.

        Args:
            db_config: Database configuration.
            schema: Database schema name (default: 'eidos').
        """
        self.db_config = db_config
        self.schema = schema
        self.table_name = "bt_trades"
        self._engine: Optional[Engine] = None

    def _get_engine(self) -> Engine:
        """Get or create database engine."""
        if self._engine is None:
            repo = DatabaseRepo(
                self.db_config, table_name=self.table_name, schema=self.schema
            )
            self._engine = repo._get_engine()
        return self._engine

    def batch_insert(
        self,
        exp_id: str,
        trades_data: List[Dict[str, Any]],
        batch_size: int = 1000,
    ) -> int:
        """
        Batch insert trade data.

        Args:
            exp_id: Experiment ID.
            trades_data: List of trade records. Each record should contain:
                - symbol: TEXT
                - deal_time: TIMESTAMPTZ
                - side: INTEGER (1=Buy, -1=Sell)
                - price: NUMERIC
                - amount: INTEGER
                - rank_at_deal: INTEGER (optional)
                - score_at_deal: FLOAT (optional)
                - reason: TEXT (optional)
                - pnl_ratio: FLOAT (optional)
                - hold_days: INTEGER (optional)
            batch_size: Batch size for insertion.

        Returns:
            Number of records inserted.
        """
        if not trades_data:
            return 0

        logger = logging.getLogger(__name__)
        logger.info(f"Inserting {len(trades_data)} trades for exp_id: {exp_id}")
        
        engine = self._get_engine()
        total_inserted = 0

        for i in range(0, len(trades_data), batch_size):
            batch = trades_data[i : i + batch_size]
            sql = text(
                f"""
                INSERT INTO "{self.schema}"."{self.table_name}"
                (exp_id, symbol, deal_time, side, price, amount, rank_at_deal, score_at_deal, reason, pnl_ratio, hold_days)
                VALUES (:exp_id, :symbol, :deal_time, :side, :price, :amount, :rank_at_deal, :score_at_deal, :reason, :pnl_ratio, :hold_days)
                ON CONFLICT (exp_id, deal_time, symbol, side) DO UPDATE SET
                    price = EXCLUDED.price,
                    amount = EXCLUDED.amount,
                    rank_at_deal = EXCLUDED.rank_at_deal,
                    score_at_deal = EXCLUDED.score_at_deal,
                    reason = EXCLUDED.reason,
                    pnl_ratio = EXCLUDED.pnl_ratio,
                    hold_days = EXCLUDED.hold_days;
                """
            )
            rows = [
                {
                    "exp_id": exp_id,
                    "symbol": record["symbol"],
                    "deal_time": record["deal_time"],
                    "side": record.get("direction", record.get("side")),
                    "price": record["price"],
                    "amount": record["amount"],
                    "rank_at_deal": record.get("rank_at_deal"),
                    "score_at_deal": record.get("score_at_deal"),
                    "reason": record.get("reason"),
                    "pnl_ratio": record.get("pnl_ratio"),
                    "hold_days": record.get("hold_days"),
                }
                for record in batch
            ]
            
            # Log first row for debugging
            if i == 0 and rows:
                logger.debug(f"First trade row exp_id: {rows[0].get('exp_id')}, symbol: {rows[0].get('symbol')}")
            
            try:
                with engine.begin() as conn:
                    result = conn.execute(sql, rows)
                    total_inserted += result.rowcount or len(rows)
            except Exception as e:
                logger.error(f"Failed to insert trades batch {i//batch_size + 1}: {e}", exc_info=True)
                logger.error(f"exp_id: {exp_id}, batch size: {len(batch)}, first row: {rows[0] if rows else 'empty'}")
                raise

        logger.info(f"Successfully inserted {total_inserted} trades for exp_id: {exp_id}")
        return total_inserted

    def get_trades(
        self,
        exp_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get trades for an experiment.

        Args:
            exp_id: Experiment ID.
            start_time: Start time (optional).
            end_time: End time (optional).
            symbol: Symbol filter (optional).

        Returns:
            List of trade records.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Getting trades for exp_id: {exp_id}, symbol: {symbol}, start_time: {start_time}, end_time: {end_time}")
        
        engine = self._get_engine()
        sql_parts = [
            f'SELECT trade_id, exp_id, symbol, deal_time, side, price, amount,',
            f'       rank_at_deal, score_at_deal, reason, pnl_ratio, hold_days, created_at',
            f'FROM "{self.schema}"."{self.table_name}"',
            f'WHERE exp_id = :exp_id'
        ]
        params = {"exp_id": exp_id}

        if start_time:
            sql_parts.append("AND deal_time >= :start_time")
            params["start_time"] = start_time
        if end_time:
            sql_parts.append("AND deal_time <= :end_time")
            params["end_time"] = end_time
        if symbol:
            sql_parts.append("AND symbol = :symbol")
            params["symbol"] = symbol

        sql_parts.append("ORDER BY deal_time ASC")
        sql = text(" ".join(sql_parts))

        with engine.connect() as conn:
            result = conn.execute(sql, params)
            # Map 'side' column to 'direction' for model compatibility
            trades = []
            for row in result:
                trade_dict = dict(row._mapping)
                if "side" in trade_dict:
                    trade_dict["direction"] = trade_dict.pop("side")
                trades.append(trade_dict)
            
            logger.info(f"Found {len(trades)} trades for exp_id: {exp_id}")
            if len(trades) == 0:
                # Debug: Check if there are any trades with this exp_id
                debug_sql = text(f'SELECT COUNT(*) as count FROM "{self.schema}"."{self.table_name}" WHERE exp_id = :exp_id')
                debug_result = conn.execute(debug_sql, {"exp_id": exp_id})
                debug_row = debug_result.fetchone()
                logger.warning(f"Debug: Total trades in DB for exp_id {exp_id}: {debug_row[0] if debug_row else 0}")
            
            return trades


class EidosModelOutputsRepo:
    """Repository for model dense outputs (core table)."""

    def __init__(self, db_config: DatabaseConfig, schema: str = "eidos"):
        """
        Initialize model outputs repository.

        Args:
            db_config: Database configuration.
            schema: Database schema name (default: 'eidos').
        """
        self.db_config = db_config
        self.schema = schema
        self.table_name = "bt_model_outputs"
        self._engine: Optional[Engine] = None

    def _get_engine(self) -> Engine:
        """Get or create database engine."""
        if self._engine is None:
            repo = DatabaseRepo(
                self.db_config, table_name=self.table_name, schema=self.schema
            )
            self._engine = repo._get_engine()
        return self._engine

    def batch_upsert(
        self,
        exp_id: str,
        outputs_data: List[Dict[str, Any]],
        batch_size: int = 1000,
    ) -> int:
        """
        Batch upsert model outputs.

        Args:
            exp_id: Experiment ID.
            outputs_data: List of output records. Each record should contain:
                - date: DATE
                - symbol: TEXT
                - score: FLOAT
                - rank: INTEGER
                - extra_scores: Dict (optional, will be stored as JSONB)
            batch_size: Batch size for insertion.

        Returns:
            Number of records inserted/updated.
        """
        if not outputs_data:
            return 0

        engine = self._get_engine()
        total_inserted = 0

        for i in range(0, len(outputs_data), batch_size):
            batch = outputs_data[i : i + batch_size]
            sql = text(
                f"""
                INSERT INTO "{self.schema}"."{self.table_name}"
                (exp_id, date, symbol, score, rank, extra_scores)
                VALUES (:exp_id, :date, :symbol, :score, :rank, :extra_scores)
                ON CONFLICT (exp_id, date, symbol) DO UPDATE SET
                    score = EXCLUDED.score,
                    rank = EXCLUDED.rank,
                    extra_scores = EXCLUDED.extra_scores;
                """
            )
            rows = [
                {
                    "exp_id": exp_id,
                    "date": record["date"],
                    "symbol": record["symbol"],
                    "score": record["score"],
                    "rank": record["rank"],
                    "extra_scores": json.dumps(record.get("extra_scores", {}))
                    if record.get("extra_scores")
                    else None,
                }
                for record in batch
            ]
            with engine.begin() as conn:
                result = conn.execute(sql, rows)
                total_inserted += result.rowcount or len(rows)

        return total_inserted

    def get_outputs(
        self,
        exp_id: str,
        date: Optional[date] = None,
        symbol: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get model outputs.

        Args:
            exp_id: Experiment ID.
            date: Specific date (optional).
            symbol: Symbol filter (optional).
            start_date: Start date (optional).
            end_date: End date (optional).
            top_k: Return only top K by rank (optional).

        Returns:
            List of output records.
        """
        engine = self._get_engine()
        sql = f"""
            SELECT exp_id, date, symbol, score, rank, extra_scores, created_at
            FROM "{self.schema}"."{self.table_name}"
            WHERE exp_id = :exp_id
        """
        params = {"exp_id": exp_id}

        if date:
            sql += " AND date = :date"
            params["date"] = date
        if start_date:
            sql += " AND date >= :start_date"
            params["start_date"] = start_date
        if end_date:
            sql += " AND date <= :end_date"
            params["end_date"] = end_date
        if symbol:
            sql += " AND symbol = :symbol"
            params["symbol"] = symbol

        if top_k:
            sql += f" ORDER BY date DESC, rank ASC LIMIT :top_k"
            params["top_k"] = top_k
        else:
            sql += " ORDER BY date DESC, rank ASC;"

        with engine.connect() as conn:
            result = conn.execute(text(sql), params)
            return [dict(row._mapping) for row in result]


class EidosModelLinksRepo:
    """Repository for model topology links (for GNN models)."""

    def __init__(self, db_config: DatabaseConfig, schema: str = "eidos"):
        """
        Initialize model links repository.

        Args:
            db_config: Database configuration.
            schema: Database schema name (default: 'eidos').
        """
        self.db_config = db_config
        self.schema = schema
        self.table_name = "bt_model_links"
        self._engine: Optional[Engine] = None

    def _get_engine(self) -> Engine:
        """Get or create database engine."""
        if self._engine is None:
            repo = DatabaseRepo(
                self.db_config, table_name=self.table_name, schema=self.schema
            )
            self._engine = repo._get_engine()
        return self._engine

    def batch_upsert(
        self,
        exp_id: str,
        links_data: List[Dict[str, Any]],
        batch_size: int = 1000,
    ) -> int:
        """
        Batch upsert model links.

        Args:
            exp_id: Experiment ID.
            links_data: List of link records. Each record should contain:
                - date: DATE
                - source: TEXT
                - target: TEXT
                - weight: FLOAT
                - link_type: TEXT (default: 'attention')
            batch_size: Batch size for insertion.

        Returns:
            Number of records inserted/updated.
        """
        if not links_data:
            return 0

        engine = self._get_engine()
        total_inserted = 0

        for i in range(0, len(links_data), batch_size):
            batch = links_data[i : i + batch_size]
            sql = text(
                f"""
                INSERT INTO "{self.schema}"."{self.table_name}"
                (exp_id, date, source, target, weight, link_type)
                VALUES (:exp_id, :date, :source, :target, :weight, :link_type)
                ON CONFLICT (exp_id, date, source, target, link_type) DO UPDATE SET
                    weight = EXCLUDED.weight;
                """
            )
            rows = [
                {
                    "exp_id": exp_id,
                    "date": record["date"],
                    "source": record["source"],
                    "target": record["target"],
                    "weight": record["weight"],
                    "link_type": record.get("link_type", "attention"),
                }
                for record in batch
            ]
            with engine.begin() as conn:
                result = conn.execute(sql, rows)
                total_inserted += result.rowcount or len(rows)

        return total_inserted

    def get_links(
        self,
        exp_id: str,
        date: Optional[date] = None,
        source: Optional[str] = None,
        target: Optional[str] = None,
        link_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get model links.

        Args:
            exp_id: Experiment ID.
            date: Specific date (optional).
            source: Source symbol filter (optional).
            target: Target symbol filter (optional).
            link_type: Link type filter (optional).
            top_k: Return only top K by weight (optional).

        Returns:
            List of link records.
        """
        engine = self._get_engine()
        sql = f"""
            SELECT exp_id, date, source, target, weight, link_type, created_at
            FROM "{self.schema}"."{self.table_name}"
            WHERE exp_id = :exp_id
        """
        params = {"exp_id": exp_id}

        if date:
            sql += " AND date = :date"
            params["date"] = date
        if source:
            sql += " AND source = :source"
            params["source"] = source
        if target:
            sql += " AND target = :target"
            params["target"] = target
        if link_type:
            sql += " AND link_type = :link_type"
            params["link_type"] = link_type

        if top_k:
            sql += f" ORDER BY weight DESC LIMIT :top_k"
            params["top_k"] = top_k
        else:
            sql += " ORDER BY date DESC, weight DESC;"

        with engine.connect() as conn:
            result = conn.execute(text(sql), params)
            return [dict(row._mapping) for row in result]


class EidosEmbeddingsRepo:
    """Repository for model embeddings."""

    def __init__(self, db_config: DatabaseConfig, schema: str = "eidos"):
        """
        Initialize embeddings repository.

        Args:
            db_config: Database configuration.
            schema: Database schema name (default: 'eidos').
        """
        self.db_config = db_config
        self.schema = schema
        self.table_name = "bt_embeddings"
        self._engine: Optional[Engine] = None

    def _get_engine(self) -> Engine:
        """Get or create database engine."""
        if self._engine is None:
            repo = DatabaseRepo(
                self.db_config, table_name=self.table_name, schema=self.schema
            )
            self._engine = repo._get_engine()
        return self._engine

    def batch_upsert(
        self,
        exp_id: str,
        embeddings_data: List[Dict[str, Any]],
        batch_size: int = 1000,
    ) -> int:
        """
        Batch upsert embeddings.

        Args:
            exp_id: Experiment ID.
            embeddings_data: List of embedding records. Each record should contain:
                - date: DATE
                - symbol: TEXT
                - vec: List[float] or numpy array (will be converted to vector)
                - vec_dim: INTEGER (vector dimension)
            batch_size: Batch size for insertion.

        Returns:
            Number of records inserted/updated.
        """
        if not embeddings_data:
            return 0

        engine = self._get_engine()
        total_inserted = 0

        for i in range(0, len(embeddings_data), batch_size):
            batch = embeddings_data[i : i + batch_size]
            sql = text(
                f"""
                INSERT INTO "{self.schema}"."{self.table_name}"
                (exp_id, date, symbol, vec, vec_dim)
                VALUES (:exp_id, :date, :symbol, :vec::vector, :vec_dim)
                ON CONFLICT (exp_id, date, symbol) DO UPDATE SET
                    vec = EXCLUDED.vec,
                    vec_dim = EXCLUDED.vec_dim;
                """
            )
            rows = []
            for record in batch:
                vec = record["vec"]
                # Convert list/numpy array to PostgreSQL vector format string
                if isinstance(vec, list):
                    vec_str = "[" + ",".join(str(float(x)) for x in vec) + "]"
                else:
                    # Assume numpy array or similar
                    import numpy as np
                    if isinstance(vec, np.ndarray):
                        vec_str = "[" + ",".join(str(float(x)) for x in vec) + "]"
                    else:
                        vec_str = str(vec)

                rows.append(
                    {
                        "exp_id": exp_id,
                        "date": record["date"],
                        "symbol": record["symbol"],
                        "vec": vec_str,
                        "vec_dim": record["vec_dim"],
                    }
                )
            with engine.begin() as conn:
                result = conn.execute(sql, rows)
                total_inserted += result.rowcount or len(rows)

        return total_inserted

    def get_embeddings(
        self,
        exp_id: str,
        date: Optional[date] = None,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get embeddings.

        Args:
            exp_id: Experiment ID.
            date: Specific date (optional).
            symbol: Symbol filter (optional).

        Returns:
            List of embedding records.
        """
        engine = self._get_engine()
        sql = f"""
            SELECT exp_id, date, symbol, vec, vec_dim, created_at
            FROM "{self.schema}"."{self.table_name}"
            WHERE exp_id = :exp_id
        """
        params = {"exp_id": exp_id}

        if date:
            sql += " AND date = :date"
            params["date"] = date
        if symbol:
            sql += " AND symbol = :symbol"
            params["symbol"] = symbol

        sql += " ORDER BY date DESC;"

        with engine.connect() as conn:
            result = conn.execute(text(sql), params)
            return [dict(row._mapping) for row in result]


class EidosRepo:
    """
    Main repository class that aggregates all EDiOS repositories.

    Provides a unified interface for all EDiOS data operations.
    """

    def __init__(self, db_config: DatabaseConfig, schema: str = "eidos"):
        """
        Initialize EDiOS repository.

        Args:
            db_config: Database configuration.
            schema: Database schema name (default: 'eidos').
        """
        self.db_config = db_config
        self.schema = schema
        self.experiment = EidosExperimentRepo(db_config, schema)
        self.ledger = EidosLedgerRepo(db_config, schema)
        self.trades = EidosTradesRepo(db_config, schema)
        self.model_outputs = EidosModelOutputsRepo(db_config, schema)
        self.model_links = EidosModelLinksRepo(db_config, schema)
        self.embeddings = EidosEmbeddingsRepo(db_config, schema)

    def save_backtest_results(
        self,
        exp_id: str,
        ledger_data: Optional[List[Dict[str, Any]]] = None,
        trades_data: Optional[List[Dict[str, Any]]] = None,
        model_outputs_data: Optional[List[Dict[str, Any]]] = None,
        model_links_data: Optional[List[Dict[str, Any]]] = None,
        embeddings_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, int]:
        """
        Save all backtest results in a transaction.

        Args:
            exp_id: Experiment ID.
            ledger_data: Ledger data (optional).
            trades_data: Trades data (optional).
            model_outputs_data: Model outputs data (optional).
            model_links_data: Model links data (optional).
            embeddings_data: Embeddings data (optional).

        Returns:
            Dictionary with counts of inserted records for each table.
        """
        logger = logging.getLogger(__name__)
        
        counts = {}

        # Save all data (each batch_* method handles its own transaction)
        if ledger_data and len(ledger_data) > 0:
            try:
                counts["ledger"] = self.ledger.batch_upsert(exp_id, ledger_data)
                logger.info(f"Saved {counts['ledger']} ledger records")
            except Exception as e:
                logger.error(f"Failed to save ledger data: {e}", exc_info=True)
                counts["ledger"] = 0
        
        if trades_data and len(trades_data) > 0:
            try:
                counts["trades"] = self.trades.batch_insert(exp_id, trades_data)
                logger.info(f"Saved {counts['trades']} trade records")
            except Exception as e:
                logger.error(f"Failed to save trades data: {e}", exc_info=True)
                counts["trades"] = 0
        
        if model_outputs_data and len(model_outputs_data) > 0:
            try:
                counts["model_outputs"] = self.model_outputs.batch_upsert(
                    exp_id, model_outputs_data
                )
                logger.info(f"Saved {counts['model_outputs']} model output records")
            except Exception as e:
                logger.error(f"Failed to save model outputs data: {e}", exc_info=True)
                counts["model_outputs"] = 0
        
        if model_links_data and len(model_links_data) > 0:
            try:
                counts["model_links"] = self.model_links.batch_upsert(
                    exp_id, model_links_data
                )
                logger.info(f"Saved {counts['model_links']} model link records")
            except Exception as e:
                logger.error(f"Failed to save model links data: {e}", exc_info=True)
                counts["model_links"] = 0
        
        if embeddings_data and len(embeddings_data) > 0:
            try:
                counts["embeddings"] = self.embeddings.batch_upsert(
                    exp_id, embeddings_data
                )
                logger.info(f"Saved {counts['embeddings']} embedding records")
            except Exception as e:
                logger.error(f"Failed to save embeddings data: {e}", exc_info=True)
                counts["embeddings"] = 0

        return counts

