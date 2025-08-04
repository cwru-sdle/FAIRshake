import gc
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, List

import tensorflow as tf

from FAIRshake.data_loading.tf_data_loader import TfDataLoader
from FAIRshake.exporting.exporter import Exporter
from FAIRshake.integration.integration import Integrator
from FAIRshake.preprocessing.preprocessing import Preprocessor
from FAIRshake.utils.logger import setup_logging
from FAIRshake.utils.resource_utils import get_max_workers


class ExecutionPipeline:
    """
    A modular pipeline that orchestrates data loading, preprocessing, integration, and exporting.
    Allows streaming access to intermediate datasets. 
    You can call each step individually or use run_all() for a complete flow.
    """

    def __init__(
        self,
        input_base_dir: str,
        output_base_dir: str,
        batch_size: int = 32,
        data_file_types: Optional[List[str]] = None,
        metadata_file_types: Optional[List[str]] = None,
        require_metadata: bool = True,
        load_metadata_files: bool = True,
        load_detector_metadata: bool = False,
        require_all_formats: bool = False,
        average_frames: bool = False,
        enable_profiling: bool = False,
        tf_data_debug_mode: bool = False,
        pattern: str = "**/*",
        log_dir: Optional[str] = None,
        log_level: int = logging.INFO,
        preprocessing_config: Optional[Dict[str, Any]] = None,
        enable_integration: bool = False,
        integration_config: Optional[Dict[str, Any]] = None,
        enable_exporting: bool = False,
        exporting_config: Optional[Dict[str, Any]] = None,
        enable_preprocessing: bool = True,
        enable_file_logging: bool = True,
    ):
        """
        Initialize the ExecutionPipeline with configuration parameters.

        For details on each argument, see the original docstring or user docs.
        """
        self.input_base_dir: Path = Path(input_base_dir)
        self.output_base_dir: Path = Path(output_base_dir)
        self.batch_size: int = batch_size

        self.data_file_types: List[str] = data_file_types or [
            ".ge2", ".tif", ".edf", ".cbf", ".mar3450", ".h5", ".png"
        ]
        self.metadata_file_types: List[str] = metadata_file_types or [
            ".json", ".poni", ".instprm", ".geom", ".spline"
        ]
        self.require_metadata: bool = require_metadata
        self.load_metadata_files: bool = load_metadata_files
        self.load_detector_metadata: bool = load_detector_metadata
        self.require_all_formats: bool = require_all_formats
        self.average_frames: bool = average_frames
        self.enable_profiling: bool = enable_profiling
        self.tf_data_debug_mode: bool = tf_data_debug_mode
        self.pattern: str = pattern
        self.log_level: int = log_level
        self.preprocessing_config: Optional[Dict[str, Any]] = preprocessing_config
        self.enable_integration: bool = enable_integration
        self.integration_config: Optional[Dict[str, Any]] = integration_config
        self.enable_exporting: bool = enable_exporting
        self.exporting_config: Optional[Dict[str, Any]] = exporting_config
        self.enable_preprocessing: bool = enable_preprocessing
        self.enable_file_logging: bool = enable_file_logging

        # Prepare logging
        if self.enable_file_logging:
            if log_dir:
                self.log_dir: Path = Path(log_dir)
            else:
                self.log_dir: Path = self.output_base_dir / "logs"
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.logger, self.log_file, self.log_records = setup_logging(
                log_dir=self.log_dir, log_level=self.log_level
            )
        else:
            self.logger, self.log_file, self.log_records = setup_logging(
                log_dir=None, log_level=self.log_level
            )

        self.logger.info("Initializing ExecutionPipeline.")
        self.setup_environment()
        self.setup_gpus()

        # Max workers
        max_workers: int = get_max_workers()
        self.logger.info(f"Max workers for the job: {max_workers}")

        # Initialize Data Loader
        self.data_loader: TfDataLoader = TfDataLoader(
            base_path=self.input_base_dir,
            require_metadata=self.require_metadata,
            load_metadata_files=self.load_metadata_files,
            load_detector_metadata=self.load_detector_metadata,
            data_file_types=self.data_file_types,
            metadata_file_types=self.metadata_file_types,
            require_all_formats=self.require_all_formats,
            average_frames=self.average_frames,
            log_dir=self.log_dir if self.enable_file_logging else None,
        )

        # Initialize Preprocessor if enabled
        if self.enable_preprocessing:
            if self.preprocessing_config:
                self.preprocessor: Preprocessor = Preprocessor(
                    base_path=self.input_base_dir,
                    dark_field_path=self.preprocessing_config.get("dark_field_path"),
                    mask_file_path=self.preprocessing_config.get("mask_file_path"),
                    invert_mask=self.preprocessing_config.get("invert_mask", False),
                    min_intensity=self.preprocessing_config.get("min_intensity"),
                    max_intensity=self.preprocessing_config.get("max_intensity"),
                )
                self.logger.info("Preprocessor initialized with config.")
            else:
                self.preprocessor: Preprocessor = Preprocessor(
                    base_path=self.input_base_dir
                )
                self.logger.info("Preprocessor initialized without config.")
        else:
            self.preprocessor = None
            self.logger.info("Preprocessing is disabled.")

        # Initialize Integrator if enabled
        if self.enable_integration:
            if not self.integration_config:
                raise ValueError("Integration is enabled but 'integration_config' is not provided.")
            poni_file_relative_path = self.integration_config.get("poni_file_path")
            if not poni_file_relative_path:
                raise ValueError("Integration is enabled but 'poni_file_path' is not provided.")

            poni_file_path = self.input_base_dir / poni_file_relative_path
            if not poni_file_path.exists():
                raise FileNotFoundError(f"The specified poni_file_path does not exist: {poni_file_path}")

            self.integration_config["poni_file_path"] = str(poni_file_path)
            self.integration_processor: Integrator = Integrator(
                integration_config=self.integration_config
            )
            self.logger.info(f"Integrator initialized: {poni_file_path}")
        else:
            self.integration_processor = None
            self.logger.info("Integration is disabled.")

        # Initialize Exporter if enabled
        if self.enable_exporting:
            if not self.exporting_config or "output_directory" not in self.exporting_config:
                raise ValueError("Exporting is enabled but 'output_directory' is not provided.")
            self.exporter: Exporter = Exporter(
                output_directory=self.exporting_config["output_directory"],
                logger=self.logger,
                options=self.exporting_config.get("options", {}),
                naming_convention=self.exporting_config.get("naming_convention", "{GE_filenumber}_{sample_id}"),
                file_format=self.exporting_config.get("file_format", "fxye"),
            )
            self.logger.info("Exporter initialized.")
        else:
            self.exporter = None
            self.logger.info("Exporting is disabled.")

        # Profiling directory setup
        if self.enable_profiling:
            self.profiler_dir: Path = self.output_base_dir / "tf_logs"
            self.profiler_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Profiler logs will be saved to {self.profiler_dir}")
        else:
            self.profiler_dir = None

        # Will hold the current tf.data.Dataset in each stage
        self.dataset: Optional[tf.data.Dataset] = None

    def setup_environment(self):
        """Set environment variables for TensorFlow to manage GPU memory and logging."""
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF logs
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    def setup_gpus(self):
        """Configure TensorFlow GPU settings to allow memory growth."""
        physical_gpus = tf.config.list_physical_devices("GPU")
        if physical_gpus:
            try:
                for gpu in physical_gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices("GPU")
                self.logger.info(
                    f"{len(physical_gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs"
                )
            except RuntimeError as e:
                self.logger.exception(f"Error setting up GPUs: {e}")
        else:
            self.logger.info("No GPU found. Running on CPU.")

    # ------------------------------------------------------
    # Modular Methods
    # ------------------------------------------------------

    def build_dataset(
        self,
        pattern: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> tf.data.Dataset:
        """
        Create the raw tf.data.Dataset from the data loader (images + metadata),
        without preprocessing or integration applied yet.
        """
        if pattern is None:
            pattern = self.pattern
        if batch_size is None:
            batch_size = self.batch_size

        self.logger.info(f"Building dataset with pattern={pattern}, batch_size={batch_size}.")
        ds = self.data_loader.create_dataset(pattern=pattern, batch_size=batch_size)
        self.dataset = ds
        return ds

    def apply_preprocessing(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Applies preprocessing transforms if enabled. Returns a new dataset.
        """
        if self.enable_preprocessing and self.preprocessor:
            self.logger.info("Applying preprocessing to dataset.")
            ds = self.preprocessor.preprocess_dataset(dataset)
            return ds
        else:
            self.logger.info("Preprocessing is disabled or no preprocessor available. Returning dataset as-is.")
            return dataset

    def apply_integration(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Applies integration transforms if enabled. Returns a new dataset.
        """
        if self.enable_integration and self.integration_processor:
            self.logger.info("Applying integration to dataset.")
            ds = self.integration_processor.integrate_dataset(dataset)
            return ds
        else:
            self.logger.info("Integration is disabled or integrator not available. Returning dataset as-is.")
            return dataset

    def export_dataset(self, dataset: tf.data.Dataset):
        """
        If enabled, consumes the dataset in streaming fashion and exports the results.
        """
        if self.enable_exporting and self.exporter:
            self.logger.info("Exporting dataset...")
            self.exporter.export_dataset(dataset)
            self.logger.info("Export completed.")
        else:
            self.logger.info("Exporter disabled or not available. Skipping export.")

    # ------------------------------------------------------
    # End-to-End Convenience Method
    # ------------------------------------------------------

    def run_all(self) -> Optional[tf.data.Dataset]:
        """
        Convenience method that:
         1) Builds the dataset
         2) Applies preprocessing
         3) Applies integration
         4) Exports (if enabled)
        It returns the final dataset. 
        If you want to do custom steps, call the individual methods instead.
        """
        self.logger.info("Starting end-to-end pipeline run.")

        try:
            # Start TF profiler if enabled
            if self.enable_profiling and self.profiler_dir is not None:
                self.logger.info(f"Starting TensorFlow profiler. Logs -> {self.profiler_dir}")
                tf.profiler.experimental.start(str(self.profiler_dir))

            ds = self.build_dataset()
            if ds is None:
                self.logger.error("Dataset creation failed.")
                return None

            ds = self.apply_preprocessing(ds)
            ds = self.apply_integration(ds)
            self.export_dataset(ds)

            self.dataset = ds
            return ds

        except Exception as e:
            self.logger.error(f"Error during pipeline run_all: {e}", exc_info=True)
            return None

        finally:
            if self.enable_profiling and self.profiler_dir is not None:
                tf.profiler.experimental.stop()
                self.logger.info(f"TensorFlow profiler stopped. Logs saved to {self.profiler_dir}")
            gc.collect()
            self.logger.info("End-to-end pipeline run completed.")

    # ------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------

    def shutdown(self):
        """
        Explicitly shut down resources such as Integrator or Exporter thread pools.
        Call this AFTER you are done iterating over the dataset or performing custom tasks.
        """
        self.logger.info("Shutting down ExecutionPipeline resources.")

        if self.enable_integration and self.integration_processor is not None:
            try:
                self.integration_processor.shutdown()
                self.logger.info("Integrator shutdown successful.")
            except Exception as e:
                self.logger.error(f"Error during Integrator shutdown: {e}", exc_info=True)

        if self.enable_exporting and self.exporter is not None:
            try:
                self.exporter.shutdown()
                self.logger.info("Exporter shutdown successful.")
            except Exception as e:
                self.logger.error(f"Error during Exporter shutdown: {e}", exc_info=True)

        self.logger.info("Pipeline resources have been shut down.")
