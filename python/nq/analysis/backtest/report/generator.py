"""
Backtest report generator.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from nq.config import DatabaseConfig

from .loader import EidosDataLoader
from .metrics.registry import MetricRegistry
from .models import BacktestData, MetricResult, ReportConfig

logger = logging.getLogger(__name__)


class BacktestReportGenerator:
    """
    Generate backtest reports from Eidos database.
    
    This class coordinates data loading, metric calculation, and report formatting.
    """
    
    def __init__(self, db_config: DatabaseConfig, schema: str = "eidos"):
        """
        Initialize report generator.
        
        Args:
            db_config: Database configuration
            schema: Database schema (default 'eidos')
        """
        self.loader = EidosDataLoader(db_config, schema)
        self.schema = schema
    
    def generate_report(
        self,
        exp_id: str,
        config: Optional[ReportConfig] = None,
    ) -> Dict:
        """
        Generate backtest report for a single experiment.
        
        Args:
            exp_id: Experiment ID
            config: Report configuration (optional)
        
        Returns:
            Dictionary containing report data
        """
        if config is None:
            config = ReportConfig()
        
        # Load data
        logger.info(f"Loading data for experiment {exp_id}")
        data = self.loader.load_experiment(exp_id)
        
        # Calculate metrics
        logger.info("Calculating metrics")
        metrics = self._calculate_metrics(data, config)
        
        # Organize metrics by category
        metrics_by_category = self._organize_by_category(metrics)
        
        # Build report
        report = {
            "exp_id": exp_id,
            "experiment_name": data.experiment.get("name", exp_id),
            "start_date": data.experiment.get("start_date", ""),
            "end_date": data.experiment.get("end_date", ""),
            "generated_at": datetime.now().isoformat(),
            "metrics": [m.to_dict() for m in metrics],
            "metrics_by_category": {
                category: [m.to_dict() for m in category_metrics]
                for category, category_metrics in metrics_by_category.items()
            },
        }
        
        logger.info(f"Generated report with {len(metrics)} metrics")
        return report
    
    def _calculate_metrics(
        self,
        data: BacktestData,
        config: ReportConfig,
    ) -> List[MetricResult]:
        """
        Calculate all metrics based on configuration.
        
        Args:
            data: Backtest data
            config: Report configuration
        
        Returns:
            List of metric results
        """
        # Get all registered metrics
        all_metrics = MetricRegistry.list_metrics()
        
        # Filter by category if specified
        if config.metric_categories:
            all_metrics = [
                m for m in all_metrics if m.category in config.metric_categories
            ]
        
        # Filter by name if specified
        if config.metric_names:
            metric_keys = set(config.metric_names)
            all_metrics = [
                m for m in all_metrics if m.get_key() in metric_keys or m.name in config.metric_names
            ]
        
        # Calculate metrics
        results = []
        for metric_def in all_metrics:
            try:
                calculator_class = metric_def.calculator
                
                # Instantiate calculator (should be a class that extends BaseMetricCalculator)
                if isinstance(calculator_class, type):
                    calculator = calculator_class()
                    result = calculator.calculate(data)
                else:
                    # It's a function, call it directly
                    result = calculator_class(data)
                
                # Ensure result is MetricResult
                if not isinstance(result, MetricResult):
                    logger.warning(f"Metric {metric_def.get_key()} did not return MetricResult")
                    continue
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to calculate metric {metric_def.get_key()}: {e}", exc_info=True)
                continue
        
        return results
    
    def _organize_by_category(
        self,
        metrics: List[MetricResult],
    ) -> Dict[str, List[MetricResult]]:
        """
        Organize metrics by category.
        
        Args:
            metrics: List of metric results
        
        Returns:
            Dictionary mapping category to list of metrics
        """
        organized = {}
        for metric in metrics:
            category = metric.category
            if category not in organized:
                organized[category] = []
            organized[category].append(metric)
        
        return organized
    
    def generate_comparison_report(
        self,
        exp_ids: List[str],
        config: Optional[ReportConfig] = None,
    ) -> Dict:
        """
        Generate comparison report for multiple experiments.
        
        Args:
            exp_ids: List of experiment IDs
            config: Report configuration (optional)
        
        Returns:
            Dictionary containing comparison report data
        """
        if config is None:
            config = ReportConfig()
        
        # Load all experiments
        logger.info(f"Loading data for {len(exp_ids)} experiments")
        all_data = self.loader.load_experiments(exp_ids)
        
        # Calculate metrics for each experiment
        all_reports = []
        for data in all_data:
            metrics = self._calculate_metrics(data, config)
            all_reports.append({
                "exp_id": data.exp_id,
                "experiment_name": data.experiment.get("name", data.exp_id),
                "metrics": {m.name: m.value for m in metrics},
            })
        
        # Build comparison report
        report = {
            "experiments": all_reports,
            "generated_at": datetime.now().isoformat(),
        }
        
        return report

