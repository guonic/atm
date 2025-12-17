"""
State repository for tracking ingestion progress.

Supports file-based and database-based state storage for checkpoint/resume functionality.
"""

import json
import logging
from abc import ABC, abstractmethod
from atm.repo.database_repo import DatabaseRepo
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from sqlalchemy import text

logger = logging.getLogger(__name__)


class IngestionState(BaseModel):
    """Ingestion state model."""

    task_name: str = Field(..., description="Task name")
    last_processed_key: Optional[str] = Field(None, description="Last processed record key (e.g., ts_code, date)")
    last_processed_time: Optional[datetime] = Field(None, description="Last processed timestamp")
    total_fetched: int = Field(0, description="Total records fetched")
    total_saved: int = Field(0, description="Total records saved")
    total_errors: int = Field(0, description="Total errors encountered")
    status: str = Field("running", description="Status: running, completed, failed, paused")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    mode: str = Field("upsert", description="Ingestion mode: upsert (覆盖更新) or append (追加)")
    lock_owner: Optional[str] = Field(None, description="Lock owner identifier (process ID or instance ID)")
    lock_acquired_at: Optional[datetime] = Field(None, description="Lock acquisition time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="State creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }


class BaseStateRepo(ABC):
    """Abstract base class for state repositories."""

    @abstractmethod
    def get_state(self, task_name: str) -> Optional[IngestionState]:
        """
        Get ingestion state for a task.

        Args:
            task_name: Task name.

        Returns:
            IngestionState if found, None otherwise.
        """
        pass

    @abstractmethod
    def save_state(self, state: IngestionState) -> bool:
        """
        Save ingestion state.

        Args:
            state: Ingestion state to save.

        Returns:
            True if save was successful.
        """
        pass

    @abstractmethod
    def delete_state(self, task_name: str) -> bool:
        """
        Delete ingestion state for a task.

        Args:
            task_name: Task name.

        Returns:
            True if deletion was successful.
        """
        pass

    @abstractmethod
    def list_states(self) -> list[IngestionState]:
        """
        List all ingestion states.

        Returns:
            List of ingestion states.
        """
        pass


class FileStateRepo(BaseStateRepo):
    """File-based state repository."""

    def __init__(self, state_dir: str = "storage/state"):
        """
        Initialize file-based state repository.

        Args:
            state_dir: Directory to store state files.
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _get_state_file(self, task_name: str) -> Path:
        """Get state file path for a task."""
        # Sanitize task name for filename
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_name)
        return self.state_dir / f"{safe_name}.json"

    def get_state(self, task_name: str) -> Optional[IngestionState]:
        """Get ingestion state from file."""
        state_file = self._get_state_file(task_name)
        if not state_file.exists():
            return None

        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Convert datetime strings back to datetime objects
                datetime_fields = ["last_processed_time", "created_at", "updated_at", "lock_acquired_at"]
                for field in datetime_fields:
                    if field in data and data[field]:
                        try:
                            data[field] = datetime.fromisoformat(data[field])
                        except (ValueError, TypeError):
                            # If parsing fails, set to None
                            data[field] = None
                return IngestionState(**data)
        except Exception as e:
            logger.error(f"Failed to load state from {state_file}: {e}")
            return None

    def save_state(self, state: IngestionState) -> bool:
        """Save ingestion state to file."""
        state_file = self._get_state_file(state.task_name)
        state.updated_at = datetime.now()

        try:
            # Use model_dump with mode='json' to serialize datetime objects
            # Or manually convert datetime to ISO format strings
            data = state.model_dump()
            
            # Convert datetime objects to ISO format strings for JSON serialization
            datetime_fields = ["last_processed_time", "created_at", "updated_at", "lock_acquired_at"]
            for field in datetime_fields:
                if field in data and data[field] is not None:
                    if isinstance(data[field], datetime):
                        data[field] = data[field].isoformat()
                    elif isinstance(data[field], str):
                        # Already a string, keep as is
                        pass
            
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to save state to {state_file}: {e}")
            return False

    def delete_state(self, task_name: str) -> bool:
        """Delete state file."""
        state_file = self._get_state_file(task_name)
        try:
            if state_file.exists():
                state_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete state file {state_file}: {e}")
            return False

    def list_states(self) -> list[IngestionState]:
        """List all states from files."""
        states = []
        for state_file in self.state_dir.glob("*.json"):
            try:
                task_name = state_file.stem
                state = self.get_state(task_name)
                if state:
                    states.append(state)
            except Exception as e:
                logger.warning(f"Failed to load state from {state_file}: {e}")
        return states


class DatabaseStateRepo(BaseStateRepo):
    """Database-based state repository."""

    def __init__(self, db_config, schema: str = "quant"):
        """
        Initialize database-based state repository.

        Args:
            db_config: Database configuration.
            schema: Database schema.
        """

        self.db_config = db_config
        self.schema = schema
        self._repo = DatabaseRepo(
            config=db_config,
            table_name="ingestion_state",
            schema=schema,
            on_conflict="update",
        )
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Ensure state table exists."""
        engine = self._repo._get_engine()
        table_name = self._repo._get_full_table_name()

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            task_name VARCHAR(255) PRIMARY KEY,
            last_processed_key VARCHAR(255),
            last_processed_time TIMESTAMP,
            total_fetched INTEGER DEFAULT 0,
            total_saved INTEGER DEFAULT 0,
            total_errors INTEGER DEFAULT 0,
            status VARCHAR(50) DEFAULT 'running',
            error_message TEXT,
            metadata JSONB DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        try:
            with engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to create state table (may already exist): {e}")

    def get_state(self, task_name: str) -> Optional[IngestionState]:
        """Get ingestion state from database."""
        engine = self._repo._get_engine()
        table_name = self._repo._get_full_table_name()

        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text(f'SELECT * FROM {table_name} WHERE task_name = :task_name'),
                    {"task_name": task_name},
                )
                row = result.fetchone()
                if row:
                    data = dict(row._mapping)
                    # Handle JSONB metadata field
                    if "metadata" in data:
                        if isinstance(data["metadata"], str):
                            data["metadata"] = json.loads(data["metadata"])
                        elif data["metadata"] is None:
                            data["metadata"] = {}
                    else:
                        data["metadata"] = {}
                    return IngestionState(**data)
        except Exception as e:
            logger.error(f"Failed to get state from database: {e}")
        return None

    def save_state(self, state: IngestionState) -> bool:
        """Save ingestion state to database."""
        state.updated_at = datetime.now()
        data = state.model_dump(exclude_none=True)
        # Convert datetime to string for database
        if "last_processed_time" in data and isinstance(data["last_processed_time"], datetime):
            data["last_processed_time"] = data["last_processed_time"].isoformat()
        if "created_at" in data and isinstance(data["created_at"], datetime):
            data["created_at"] = data["created_at"].isoformat()
        if "updated_at" in data and isinstance(data["updated_at"], datetime):
            data["updated_at"] = data["updated_at"].isoformat()
        # Convert metadata dict to JSON string for JSONB field
        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = json.dumps(data["metadata"])

        try:
            return self._repo.save(data)
        except Exception as e:
            logger.error(f"Failed to save state to database: {e}")
            return False

    def delete_state(self, task_name: str) -> bool:
        """Delete state from database."""
        engine = self._repo._get_engine()
        table_name = self._repo._get_full_table_name()

        try:
            with engine.connect() as conn:
                conn.execute(
                    text(f'DELETE FROM {table_name} WHERE task_name = :task_name'),
                    {"task_name": task_name},
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete state from database: {e}")
            return False

    def list_states(self) -> list[IngestionState]:
        """List all states from database."""
        engine = self._repo._get_engine()
        table_name = self._repo._get_full_table_name()

        try:
            with engine.connect() as conn:
                result = conn.execute(text(f'SELECT * FROM {table_name}'))
                states = []
                for row in result:
                    data = dict(row._mapping)
                    # Handle JSONB metadata field
                    if "metadata" in data and isinstance(data["metadata"], str):
                        data["metadata"] = json.loads(data["metadata"])
                    states.append(IngestionState(**data))
                return states
        except Exception as e:
            logger.error(f"Failed to list states from database: {e}")
            return []

    def close(self) -> None:
        """Close database connection."""
        if self._repo:
            self._repo.close()

