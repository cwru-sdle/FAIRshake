# FAIRshake/__init__.py

# Import classes and functions from modules within the FAIRshake package
from .execution_pipeline.pipeline import ExecutionPipeline
from .data_loading.file_loader import FileLoader
from .data_loading.tf_data_loader import TfDataLoader
from .utils.logger import setup_logging
from .integration.integration import Integrator
from .exporting.exporter import Exporter
from .exporting.file_writers.fxye_writer import FxyeWriter
from .exporting.file_writers.xye_writer import XyeWriter
from .exporting.file_writers.xy_writer import XyWriter
from .exporting.file_writers.tiff_writer import TiffWriter
from .data_handling.data_handler import DataHandler
from .utils.resource_utils import get_max_workers
from .benchmarking import benchmark  # Import the benchmark function

# Define __all__ for explicit export of module components
__all__ = [
    "ExecutionPipeline",
    "FileLoader",
    "TfDataLoader",
    "setup_logging",
    "Integrator",
    "Exporter",
    "FxyeWriter",
    "XyeWriter",
    "XyWriter",
    "TiffWriter",
    "DataHandler",
    "get_max_workers",
    "benchmark",  # Add 'benchmark' to __all__
]
