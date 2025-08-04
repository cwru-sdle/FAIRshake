# FAIRshake/utils/__init__.py

from .logger import setup_logging
from .resource_utils import get_max_workers

__all__ = ["setup_logging", "get_max_workers"]
