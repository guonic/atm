"""
Database repository implementation using SQLAlchemy.

Stores data in PostgreSQL/TimescaleDB database.
"""

from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from atm.config import DatabaseConfig
from atm.repo.base import BaseRepo, ConnectionError, RepoError, SaveError


class DatabaseRepo(BaseRepo):
    """
    Database repository for storing data in PostgreSQL/TimescaleDB.

    Supports:
    - Batch inserts
    - Conflict resolution (update, ignore, error)
    - Automatic table creation (optional)
    - Connection pooling
    """

    def __init__(
        self,
        config: DatabaseConfig,
        table_name: str,
        schema: Optional[str] = None,
        on_conflict: str = "update",
        mode: str = "upsert",
    ):
        """
        Initialize database repository.

        Args:
            config: Database configuration.
            table_name: Target table name.
            schema: Database schema (defaults to config.schema).
            on_conflict: Conflict resolution strategy ('update', 'ignore', 'error').
            mode: Ingestion mode ('upsert' for 覆盖更新, 'append' for 追加).
        """
        self.config = config
        self.table_name = table_name
        self.schema = schema or config.schema
        self.on_conflict = on_conflict
        self.mode = mode  # "upsert" (覆盖更新) or "append" (追加)
        self.engine: Optional[Engine] = None
        self._connection_string = self._build_connection_string()

    def _build_connection_string(self) -> str:
        """
        Build SQLAlchemy connection string.

        Returns:
            Connection string.
        """
        return (
            f"postgresql://{self.config.user}:{self.config.password}"
            f"@{self.config.host}:{self.config.port}/{self.config.database}"
        )

    def _get_engine(self) -> Engine:
        """
        Get or create database engine.

        Returns:
            SQLAlchemy engine.
        """
        if self.engine is None:
            try:
                self.engine = create_engine(
                    self._connection_string,
                    pool_pre_ping=True,
                    pool_size=5,
                    max_overflow=10,
                )
            except Exception as e:
                raise ConnectionError(f"Failed to create database engine: {e}") from e
        return self.engine

    def _get_full_table_name(self) -> str:
        """
        Get full table name with schema.

        Returns:
            Full table name (schema.table).
        """
        return f'"{self.schema}"."{self.table_name}"' if self.schema else f'"{self.table_name}"'

    def save(self, data: Dict[str, Any]) -> bool:
        """
        Save a single data record.

        Args:
            data: Data record to save.

        Returns:
            True if save was successful.

        Raises:
            SaveError: If save operation fails.
        """
        return self.save_batch([data]) == 1

    def save_batch(self, data: List[Dict[str, Any]]) -> int:
        """
        Save multiple data records in a batch.

        Args:
            data: List of data records to save.

        Returns:
            Number of records successfully saved.

        Raises:
            SaveError: If batch save operation fails.
        """
        if not data:
            return 0

        engine = self._get_engine()
        table_name = self._get_full_table_name()

        try:
            with engine.connect() as conn:
                # Get table columns
                inspector = inspect(engine)
                columns = [col["name"] for col in inspector.get_columns(self.table_name, schema=self.schema)]

                # Filter data to match table columns and remove None values
                filtered_data = []
                for record in data:
                    filtered_record = {k: v for k, v in record.items() if k in columns and v is not None}
                    if filtered_record:
                        filtered_data.append(filtered_record)

                if not filtered_data:
                    return 0

                # Ensure all records have the same keys for batch insert
                # Get union of all keys from all records
                all_keys = set()
                for record in filtered_data:
                    all_keys.update(record.keys())
                
                # Normalize all records to have the same keys
                # For fields that are None in some records but present in others,
                # we need to ensure all records have the same structure
                # Strategy: Only include fields that are present in ALL records (or handle None explicitly)
                # But for optional fields like pretrade_date, we should exclude None values entirely
                
                # First pass: collect all keys that appear in at least one record
                all_keys = set()
                for record in filtered_data:
                    all_keys.update(record.keys())
                
                # Second pass: normalize records - only include keys that have non-None values
                # This ensures SQLAlchemy batch insert works correctly
                normalized_data = []
                for record in filtered_data:
                    # Only include keys that exist in this record and are not None
                    normalized_record = {k: v for k, v in record.items() if v is not None}
                    if normalized_record:
                        normalized_data.append(normalized_record)
                
                if not normalized_data:
                    return 0
                
                # Third pass: ensure all records have the same keys for SQLAlchemy batch insert
                # Find the intersection of keys (keys present in ALL records)
                # OR use union but pad missing keys (but SQLAlchemy doesn't support this well)
                # Better approach: Group records by their key sets and process separately
                # OR: Use the union of all keys, but only include keys that exist in each record
                
                # For SQLAlchemy batch insert, we need all records to have the same keys
                # So we'll use the union of all keys, but only include keys that exist in each record
                final_all_keys = set()
                for record in normalized_data:
                    final_all_keys.update(record.keys())
                
                # Pad all records with missing keys (set to None, but we'll exclude None in SQL)
                # Actually, SQLAlchemy's execute() with a list of dicts requires all dicts to have the same keys
                # So we need to ensure all records have the same structure
                # But if we include None values, SQLAlchemy will try to bind them
                # Solution: Only include keys that are present in ALL records, OR process in groups
                
                # Check if all records have the same keys
                first_record_keys = set(normalized_data[0].keys())
                all_same_keys = all(set(r.keys()) == first_record_keys for r in normalized_data)
                
                if not all_same_keys:
                    # Records have different keys - need to handle this
                    # Option 1: Process in groups by key set (slower but safer)
                    # Option 2: Use only common keys (may lose data)
                    # Option 3: Pad records with None and handle in SQL (but SQLAlchemy doesn't like this)
                    # Best: Process records with same keys together
                    from collections import defaultdict
                    groups = defaultdict(list)
                    for idx, record in enumerate(normalized_data):
                        key_tuple = tuple(sorted(record.keys()))
                        groups[key_tuple].append((idx, record))
                    
                    # Process each group separately
                    total_saved = 0
                    for key_tuple, group_records in groups.items():
                        group_data = [r for _, r in group_records]
                        group_keys = set(key_tuple)
                        
                        # Build SQL for this group
                        if self.on_conflict == "update":
                            columns_str = ", ".join([f'"{col}"' for col in sorted(group_keys)])
                            values_placeholders = ", ".join([f":{col}" for col in sorted(group_keys)])
                            update_set = ", ".join([f'"{col}" = EXCLUDED."{col}"' for col in sorted(group_keys)])
                            
                            pk_constraint = inspector.get_pk_constraint(self.table_name, schema=self.schema)
                            if pk_constraint and pk_constraint.get("constrained_columns"):
                                conflict_target = ", ".join([f'"{col}"' for col in pk_constraint["constrained_columns"]])
                            else:
                                conflict_target = f'"{sorted(group_keys)[0]}"'
                            
                            sql = text(
                                f"""
                                INSERT INTO {table_name} ({columns_str})
                                VALUES ({values_placeholders})
                                ON CONFLICT ({conflict_target}) DO UPDATE SET {update_set}
                                """
                            )
                        elif self.on_conflict == "ignore":
                            columns_str = ", ".join([f'"{col}"' for col in sorted(group_keys)])
                            values_placeholders = ", ".join([f":{col}" for col in sorted(group_keys)])
                            pk_constraint = inspector.get_pk_constraint(self.table_name, schema=self.schema)
                            if pk_constraint and pk_constraint.get("constrained_columns"):
                                conflict_target = ", ".join([f'"{col}"' for col in pk_constraint["constrained_columns"]])
                            else:
                                conflict_target = f'"{sorted(group_keys)[0]}"'
                            
                            sql = text(
                                f"""
                                INSERT INTO {table_name} ({columns_str})
                                VALUES ({values_placeholders})
                                ON CONFLICT ({conflict_target}) DO NOTHING
                                """
                            )
                        elif self.mode == "append":
                            columns_str = ", ".join([f'"{col}"' for col in sorted(group_keys)])
                            values_placeholders = ", ".join([f":{col}" for col in sorted(group_keys)])
                            sql = text(f"INSERT INTO {table_name} ({columns_str}) VALUES ({values_placeholders})")
                        else:
                            columns_str = ", ".join([f'"{col}"' for col in sorted(group_keys)])
                            values_placeholders = ", ".join([f":{col}" for col in sorted(group_keys)])
                            sql = text(f"INSERT INTO {table_name} ({columns_str}) VALUES ({values_placeholders})")
                        
                        result = conn.execute(sql, group_data)
                        total_saved += result.rowcount or len(group_data)
                    
                    conn.commit()
                    return total_saved
                
                # All records have the same keys - proceed normally
                all_keys = final_all_keys
                
                # Build insert statement
                if self.on_conflict == "update":
                    # PostgreSQL UPSERT using ON CONFLICT
                    # Note: Requires a unique constraint or primary key
                    columns_str = ", ".join([f'"{col}"' for col in sorted(all_keys)])
                    values_placeholders = ", ".join([f":{col}" for col in sorted(all_keys)])
                    update_set = ", ".join([f'"{col}" = EXCLUDED."{col}"' for col in sorted(all_keys)])

                    # Try to get primary key or unique constraint
                    pk_constraint = inspector.get_pk_constraint(self.table_name, schema=self.schema)
                    if pk_constraint and pk_constraint.get("constrained_columns"):
                        conflict_target = ", ".join([f'"{col}"' for col in pk_constraint["constrained_columns"]])
                    else:
                        # Fallback: use first column as conflict target
                        conflict_target = f'"{sorted(all_keys)[0]}"'

                    sql = text(
                        f"""
                        INSERT INTO {table_name} ({columns_str})
                        VALUES ({values_placeholders})
                        ON CONFLICT ({conflict_target}) DO UPDATE SET {update_set}
                        """
                    )
                elif self.on_conflict == "ignore":
                    # PostgreSQL INSERT ... ON CONFLICT DO NOTHING
                    columns_str = ", ".join([f'"{col}"' for col in sorted(all_keys)])
                    values_placeholders = ", ".join([f":{col}" for col in sorted(all_keys)])
                    pk_constraint = inspector.get_pk_constraint(self.table_name, schema=self.schema)
                    if pk_constraint and pk_constraint.get("constrained_columns"):
                        conflict_target = ", ".join([f'"{col}"' for col in pk_constraint["constrained_columns"]])
                    else:
                        conflict_target = f'"{sorted(all_keys)[0]}"'

                    sql = text(
                        f"""
                        INSERT INTO {table_name} ({columns_str})
                        VALUES ({values_placeholders})
                        ON CONFLICT ({conflict_target}) DO NOTHING
                        """
                    )
                elif self.mode == "append":
                    # Append mode: simple INSERT (no conflict handling)
                    columns_str = ", ".join([f'"{col}"' for col in sorted(all_keys)])
                    values_placeholders = ", ".join([f":{col}" for col in sorted(all_keys)])
                    sql = text(f"INSERT INTO {table_name} ({columns_str}) VALUES ({values_placeholders})")
                else:  # error
                    # Simple INSERT (will fail on conflict)
                    columns_str = ", ".join([f'"{col}"' for col in sorted(all_keys)])
                    values_placeholders = ", ".join([f":{col}" for col in sorted(all_keys)])
                    sql = text(f"INSERT INTO {table_name} ({columns_str}) VALUES ({values_placeholders})")

                # Execute batch insert with normalized data
                result = conn.execute(sql, normalized_data)
                conn.commit()
                return result.rowcount or len(filtered_data)

        except SQLAlchemyError as e:
            raise SaveError(f"Failed to save data to {table_name}: {e}") from e
        except Exception as e:
            raise SaveError(f"Unexpected error saving data: {e}") from e

    def exists(self, data: Dict[str, Any]) -> bool:
        """
        Check if a record already exists.

        This is a simple implementation that checks based on the first key-value pair.
        For production use, this should be customized based on unique constraints.

        Args:
            data: Data record to check.

        Returns:
            True if record exists, False otherwise.
        """
        if not data:
            return False

        engine = self._get_engine()
        table_name = self._get_full_table_name()

        try:
            # Use first key-value pair for existence check
            key = list(data.keys())[0]
            value = data[key]

            with engine.connect() as conn:
                sql = text(f'SELECT 1 FROM {table_name} WHERE "{key}" = :value LIMIT 1')
                result = conn.execute(sql, {"value": value})
                return result.fetchone() is not None
        except Exception:
            return False

    def close(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None

