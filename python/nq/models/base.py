"""
Base model classes for ATM project.

Defines common data models and base classes.
"""

from typing import Any, Dict, Optional
from datetime import datetime

from pydantic import BaseModel as PydanticBaseModel, Field


class BaseModel(PydanticBaseModel):
    """Base model class with common fields for ATM project."""

    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Update timestamp")

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        validate_assignment = True
