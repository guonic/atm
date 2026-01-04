"""
FastAPI dependencies for Eidos API.
"""

from functools import lru_cache

from nq.config import DatabaseConfig, load_config
from nq.repo.eidos_repo import EidosRepo


@lru_cache()
def get_db_config() -> DatabaseConfig:
    """
    Get database configuration (cached).
    
    Returns:
        Database configuration.
    """
    config = load_config()
    return config.database


def get_eidos_repo() -> EidosRepo:
    """
    Get Eidos repository instance.
    
    Returns:
        Eidos repository instance.
    """
    db_config = get_db_config()
    return EidosRepo(db_config=db_config, schema="eidos")

