"""
FastAPI dependencies for Eidos API.
"""

from functools import lru_cache

from nq.config import DatabaseConfig, load_config
from nq.repo.eidos_repo import EidosRepo
from nq.api.rest.eidos.services.kline_service import KlineService


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


def get_kline_service() -> KlineService:
    """
    Get K-line service instance.
    
    Returns:
        K-line service instance.
    """
    eidos_repo = get_eidos_repo()
    db_config = get_db_config()
    return KlineService(eidos_repo=eidos_repo, db_config=db_config)
