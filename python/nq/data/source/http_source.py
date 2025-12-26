"""
HTTP data source implementation.

Fetches data from HTTP/HTTPS endpoints (REST APIs, web scraping, etc.).
"""

import time
from typing import Any, Dict, Iterator, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from nq.data.source.base import (
    BaseSource,
    ConnectionError,
    DataFetchError,
    SourceConfig,
    SourceError,
)


class HttpSourceConfig(SourceConfig):
    """HTTP source configuration."""

    url: str
    method: str = "GET"
    headers: Dict[str, str] = {}
    params: Dict[str, Any] = {}
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 1
    data_path: Optional[str] = None  # JSONPath or key path to extract data array
    pagination: Optional[Dict[str, Any]] = None  # Pagination configuration


class HttpSource(BaseSource):
    """
    HTTP data source for fetching data from REST APIs or web endpoints.

    Supports:
    - GET and POST requests
    - Custom headers and parameters
    - Retry logic
    - Pagination
    - JSON data extraction
    """

    def __init__(self, config: HttpSourceConfig):
        """
        Initialize HTTP source.

        Args:
            config: HTTP source configuration.
        """
        super().__init__(config)
        self.config: HttpSourceConfig = config
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """
        Create a requests session with retry strategy.

        Returns:
            Configured requests session.
        """
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.retry_count,
            backoff_factor=self.config.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def test_connection(self) -> bool:
        """
        Test connection to the HTTP endpoint.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            response = self.session.head(
                self.config.url,
                headers=self.config.headers,
                timeout=self.config.timeout,
            )
            return response.status_code < 400
        except Exception:
            return False

    def fetch(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Fetch data from HTTP endpoint.

        Args:
            **kwargs: Additional parameters (can override config params).

        Yields:
            Dictionary representing a single data record.

        Raises:
            ConnectionError: If connection fails.
            DataFetchError: If data fetching fails.
        """
        # Merge kwargs with config params
        params = {**self.config.params, **kwargs}

        try:
            if self.config.pagination:
                # Handle paginated requests
                yield from self._fetch_paginated(params)
            else:
                # Single request
                yield from self._fetch_single(params)
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to {self.config.url}: {e}") from e
        except Exception as e:
            raise DataFetchError(f"Failed to fetch data: {e}") from e

    def _fetch_single(self, params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Fetch data from a single request.

        Args:
            params: Request parameters.

        Yields:
            Data records.
        """
        response = self.session.request(
            method=self.config.method,
            url=self.config.url,
            headers=self.config.headers,
            params=params if self.config.method == "GET" else None,
            json=params if self.config.method == "POST" else None,
            timeout=self.config.timeout,
        )
        response.raise_for_status()

        data = response.json()

        # Extract data array if data_path is specified
        if self.config.data_path:
            data = self._extract_data_path(data, self.config.data_path)

        # Yield records
        if isinstance(data, list):
            for record in data:
                yield record
        elif isinstance(data, dict):
            yield data
        else:
            raise DataFetchError(f"Unexpected data type: {type(data)}")

    def _fetch_paginated(self, params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Fetch data with pagination support.

        Args:
            params: Request parameters.

        Yields:
            Data records from all pages.
        """
        pagination = self.config.pagination
        page_param = pagination.get("page_param", "page")
        page_size_param = pagination.get("page_size_param", "page_size")
        page_size = pagination.get("page_size", 100)
        max_pages = pagination.get("max_pages")

        page = pagination.get("start_page", 1)
        total_fetched = 0

        while True:
            # Update params with current page
            current_params = {**params, page_param: page, page_size_param: page_size}

            # Fetch current page
            records = list(self._fetch_single(current_params))
            if not records:
                break

            # Yield records
            for record in records:
                yield record
                total_fetched += 1

            # Check if we should continue
            if len(records) < page_size:
                break  # Last page

            if max_pages and page >= max_pages:
                break

            page += 1

            # Rate limiting
            if pagination.get("delay"):
                time.sleep(pagination["delay"])

    def _extract_data_path(self, data: Any, path: str) -> Any:
        """
        Extract data using a simple path notation (e.g., "data.items").

        Args:
            data: Data to extract from.
            path: Path to extract (dot-separated keys).

        Returns:
            Extracted data.
        """
        keys = path.split(".")
        result = data
        for key in keys:
            if isinstance(result, dict):
                result = result.get(key)
            elif isinstance(result, list) and key.isdigit():
                result = result[int(key)]
            else:
                raise DataFetchError(f"Cannot extract path '{path}' from data")
            if result is None:
                break
        return result

    def close(self) -> None:
        """Close the HTTP session."""
        if self.session:
            self.session.close()

