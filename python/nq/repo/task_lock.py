"""
Task lock mechanism for preventing concurrent execution of the same task.

Supports file-based and database-based locking.
"""

import fcntl
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from nq.repo.state_repo import IngestionState

logger = logging.getLogger(__name__)


class TaskLockError(Exception):
    """Exception raised when task lock operations fail."""

    pass


class BaseTaskLock(ABC):
    """Abstract base class for task locks."""

    @abstractmethod
    def acquire(self, task_name: str, timeout: float = 0) -> bool:
        """
        Acquire lock for a task.

        Args:
            task_name: Task name.
            timeout: Maximum time to wait for lock (0 = no wait, fail immediately).

        Returns:
            True if lock was acquired, False otherwise.

        Raises:
            TaskLockError: If lock acquisition fails unexpectedly.
        """
        pass

    @abstractmethod
    def release(self, task_name: str) -> bool:
        """
        Release lock for a task.

        Args:
            task_name: Task name.

        Returns:
            True if lock was released successfully.
        """
        pass

    @abstractmethod
    def is_locked(self, task_name: str) -> bool:
        """
        Check if a task is currently locked.

        Args:
            task_name: Task name.

        Returns:
            True if task is locked, False otherwise.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass


class FileTaskLock(BaseTaskLock):
    """File-based task lock using fcntl."""

    def __init__(self, lock_dir: str = "storage/locks"):
        """
        Initialize file-based task lock.

        Args:
            lock_dir: Directory to store lock files.
        """
        self.lock_dir = Path(lock_dir)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, int] = {}  # task_name -> file descriptor

    def _get_lock_file(self, task_name: str) -> Path:
        """Get lock file path for a task."""
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_name)
        return self.lock_dir / f"{safe_name}.lock"

    def acquire(self, task_name: str, timeout: float = 0) -> bool:
        """Acquire file lock."""
        lock_file = self._get_lock_file(task_name)

        try:
            # Open lock file
            fd = os.open(str(lock_file), os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
            start_time = time.time()

            while True:
                try:
                    # Try to acquire exclusive lock (non-blocking)
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # Lock acquired successfully
                    self._locks[task_name] = fd
                    logger.debug(f"Acquired lock for task: {task_name}")
                    return True
                except BlockingIOError:
                    # Lock is held by another process
                    if timeout > 0 and (time.time() - start_time) < timeout:
                        time.sleep(0.1)  # Wait 100ms and retry
                        continue
                    else:
                        os.close(fd)
                        logger.warning(f"Failed to acquire lock for task: {task_name} (timeout or already locked)")
                        return False

        except Exception as e:
            logger.error(f"Failed to acquire lock for task {task_name}: {e}")
            if task_name in self._locks:
                try:
                    os.close(self._locks[task_name])
                except Exception:
                    pass
                del self._locks[task_name]
            raise TaskLockError(f"Failed to acquire lock: {e}") from e

    def release(self, task_name: str) -> bool:
        """Release file lock."""
        if task_name not in self._locks:
            return False

        try:
            fd = self._locks[task_name]
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            del self._locks[task_name]

            # Remove lock file
            lock_file = self._get_lock_file(task_name)
            if lock_file.exists():
                lock_file.unlink()

            logger.debug(f"Released lock for task: {task_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to release lock for task {task_name}: {e}")
            return False

    def is_locked(self, task_name: str) -> bool:
        """Check if task is locked."""
        lock_file = self._get_lock_file(task_name)
        if not lock_file.exists():
            return False

        try:
            # Try to acquire lock (non-blocking)
            fd = os.open(str(lock_file), os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Lock acquired, means it wasn't locked
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                return False
            except BlockingIOError:
                # Lock is held
                os.close(fd)
                return True
        except Exception:
            return False

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release all locks on exit."""
        for task_name in list(self._locks.keys()):
            self.release(task_name)


class DatabaseTaskLock(BaseTaskLock):
    """Database-based task lock using state repository."""

    def __init__(self, state_repo):
        """
        Initialize database-based task lock.

        Args:
            state_repo: State repository instance (DatabaseStateRepo).
        """
        self.state_repo = state_repo
        self.lock_owner = f"{os.getpid()}_{id(self)}"
        self.lock_timeout = timedelta(minutes=30)  # Lock expires after 30 minutes

    def acquire(self, task_name: str, timeout: float = 0) -> bool:
        """Acquire database lock."""
        state = self.state_repo.get_state(task_name)

        if state is None:
            # No existing state, create new one with lock
            state = IngestionState(
                task_name=task_name,
                status="running",
                lock_owner=self.lock_owner,
                lock_acquired_at=datetime.now(),
            )
            return self.state_repo.save_state(state)

        # Check if task is already locked
        if state.lock_owner and state.lock_owner != self.lock_owner:
            # Check if lock has expired
            if state.lock_acquired_at:
                lock_age = datetime.now() - state.lock_acquired_at
                if lock_age < self.lock_timeout:
                    logger.warning(
                        f"Task {task_name} is locked by {state.lock_owner} "
                        f"(acquired {lock_age} ago)"
                    )
                    return False
                else:
                    logger.warning(
                        f"Task {task_name} lock expired, acquiring new lock"
                    )

        # Acquire lock
        state.lock_owner = self.lock_owner
        state.lock_acquired_at = datetime.now()
        state.status = "running"
        return self.state_repo.save_state(state)

    def release(self, task_name: str) -> bool:
        """Release database lock."""
        state = self.state_repo.get_state(task_name)
        if state is None:
            return True  # No state, nothing to release

        # If lock_owner is None or matches, release it
        if state.lock_owner and state.lock_owner != self.lock_owner:
            logger.warning(f"Cannot release lock for task {task_name}: owned by {state.lock_owner}, current owner: {self.lock_owner}")
            return False

        # Release lock (even if lock_owner is None, clear it anyway)
        state.lock_owner = None
        state.lock_acquired_at = None
        return self.state_repo.save_state(state)

    def is_locked(self, task_name: str) -> bool:
        """Check if task is locked."""
        state = self.state_repo.get_state(task_name)
        if state is None:
            return False

        if not state.lock_owner:
            return False

        # Check if lock has expired
        if state.lock_acquired_at:
            lock_age = datetime.now() - state.lock_acquired_at
            if lock_age >= self.lock_timeout:
                return False  # Lock expired

        return True

