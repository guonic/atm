"""
Data loader for backtest report system.
Uses EidosRepo to load data from database.
"""

import logging
from typing import List

import pandas as pd

from nq.config import DatabaseConfig
from nq.repo.eidos_repo import EidosRepo

from .models import BacktestData

logger = logging.getLogger(__name__)


class EidosDataLoader:
    """
    Load backtest data from Eidos database.
    
    Directly uses EidosRepo to reuse existing data access layer.
    """
    
    def __init__(self, db_config: DatabaseConfig, schema: str = "eidos"):
        """
        Initialize data loader.
        
        Args:
            db_config: Database configuration (same as EidosRepo uses)
            schema: Database schema (default 'eidos')
        """
        # Directly use EidosRepo, reuse existing infrastructure
        self.repo = EidosRepo(db_config, schema)
        self.schema = schema
    
    def load_experiment(self, exp_id: str) -> BacktestData:
        """
        Load single experiment data.
        
        Uses EidosRepo's sub-repositories to load data, consistent with existing REST API handlers.
        
        Args:
            exp_id: Experiment ID.
        
        Returns:
            BacktestData object.
        
        Raises:
            ValueError: If experiment not found.
        """
        # Load experiment metadata (using experiment repository)
        experiment = self.repo.experiment.get_experiment(exp_id)
        if not experiment:
            raise ValueError(f"Experiment {exp_id} not found")
        
        # Load ledger data (using ledger repository)
        ledger_data = self.repo.ledger.get_ledger(exp_id)
        if ledger_data:
            ledger = pd.DataFrame(ledger_data)
            # Ensure date column is datetime
            if 'date' in ledger.columns:
                ledger['date'] = pd.to_datetime(ledger['date'])
                ledger = ledger.sort_values('date').reset_index(drop=True)
        else:
            ledger = pd.DataFrame()
        
        # Load trades data (using trades repository)
        trades_data = self.repo.trades.get_trades(exp_id)
        if trades_data:
            trades = pd.DataFrame(trades_data)
            # Ensure deal_time column is datetime
            if 'deal_time' in trades.columns:
                trades['deal_time'] = pd.to_datetime(trades['deal_time'])
        else:
            trades = pd.DataFrame()
        
        # Load model outputs (optional, using model_outputs repository)
        try:
            model_outputs_data = self.repo.model_outputs.get_model_outputs(exp_id)
            if model_outputs_data:
                model_outputs = pd.DataFrame(model_outputs_data)
                if 'date' in model_outputs.columns:
                    model_outputs['date'] = pd.to_datetime(model_outputs['date'])
            else:
                model_outputs = pd.DataFrame()
        except Exception as e:
            logger.debug(f"Could not load model outputs: {e}")
            model_outputs = pd.DataFrame()
        
        # Load model links (optional, using model_links repository)
        try:
            model_links_data = self.repo.model_links.get_model_links(exp_id)
            if model_links_data:
                model_links = pd.DataFrame(model_links_data)
                if 'date' in model_links.columns:
                    model_links['date'] = pd.to_datetime(model_links['date'])
            else:
                model_links = pd.DataFrame()
        except Exception as e:
            logger.debug(f"Could not load model links: {e}")
            model_links = pd.DataFrame()
        
        # Load embeddings (optional, using embeddings repository)
        try:
            embeddings_data = self.repo.embeddings.get_embeddings(exp_id)
            if embeddings_data:
                embeddings = pd.DataFrame(embeddings_data)
                if 'date' in embeddings.columns:
                    embeddings['date'] = pd.to_datetime(embeddings['date'])
            else:
                embeddings = pd.DataFrame()
        except Exception as e:
            logger.debug(f"Could not load embeddings: {e}")
            embeddings = pd.DataFrame()
        
        logger.info(
            f"Loaded experiment {exp_id}: "
            f"ledger={len(ledger)} rows, "
            f"trades={len(trades)} rows, "
            f"model_outputs={len(model_outputs)} rows"
        )
        
        return BacktestData(
            exp_id=exp_id,
            experiment=experiment,
            ledger=ledger,
            trades=trades,
            model_outputs=model_outputs,
            model_links=model_links,
            embeddings=embeddings,
        )
    
    def load_experiments(self, exp_ids: List[str]) -> List[BacktestData]:
        """
        Load multiple experiments data.
        
        Args:
            exp_ids: List of experiment IDs.
        
        Returns:
            List of BacktestData objects.
        """
        return [self.load_experiment(exp_id) for exp_id in exp_ids]

