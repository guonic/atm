"""
Example usage of the data ingestor system.

This example demonstrates how to use the data ingestor service to fetch and store data.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nq.config import load_config
from nq.service import DataIngestorService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Run data ingestion example."""
    # Load configuration
    config_path = Path(__file__).parent.parent.parent / "config" / "data_ingestor.yaml"
    config = load_config(str(config_path))

    # Process each enabled task
    for task_config in config.tasks:
        if not task_config.enabled:
            logger.info(f"Skipping disabled task: {task_config.name}")
            continue

        logger.info(f"Processing task: {task_config.name}")

        try:
            # Create data ingestor service
            service = DataIngestorService(
                task_config=task_config,
                db_config=config.database,
            )

            # Execute ingestion
            # You can pass additional parameters here, e.g., date range
            stats = service.ingest(
                # date="2024-01-01",  # Example parameter
            )

            # Print results
            logger.info(
                f"Task '{task_config.name}' completed: "
                f"Fetched={stats['fetched']}, "
                f"Saved={stats['saved']}, "
                f"Errors={stats['errors']}"
            )

            # Clean up
            service.close()

        except Exception as e:
            logger.error(f"Task '{task_config.name}' failed: {e}", exc_info=True)
            continue


if __name__ == "__main__":
    main()

