# FAIRshake/exporting/exporter.py

import re
import gc
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict
from typing import Any, Dict, Optional, List

import numpy as np
import tensorflow as tf

from FAIRshake.data_handling.data_handler import DataHandler
from FAIRshake.exporting.file_writers.fxye_writer import FxyeWriter
from FAIRshake.exporting.file_writers.xye_writer import XyeWriter
from FAIRshake.exporting.file_writers.xy_writer import XyWriter
from FAIRshake.exporting.file_writers.tiff_writer import TiffWriter
from FAIRshake.utils.resource_utils import get_max_workers


class Exporter:
    """
    A class for exporting data to various file formats with protections to ensure data integrity.
    """

    def __init__(
        self,
        output_directory: str,
        logger: logging.Logger,
        options: Optional[Dict[str, Any]] = None,
        naming_convention: str = "batch{batch_num}_sample{sample_num}_iter{iter}",
        file_format: str = "fxye",
    ):
        """
        Initialize the Exporter with protections based on the file format.

        Parameters
        ----------
        output_directory : str
            Directory where exported files will be saved.
        logger : logging.Logger
            Logger instance for logging messages.
        options : dict, optional
            Exporting options. Default is {"do_remove_nan": True}.
        naming_convention : str, optional
            Naming convention for exported files with placeholders for metadata.
            Example: "batch{batch_num}_sample{sample_num}_iter{iter}".
        file_format : str, optional
            Desired file format for export. Supported formats include "fxye", "xye", "xy", "tiff".
        """
        self.output_directory: Path = Path(output_directory)
        self.logger: logging.Logger = logger
        self.options: Dict[str, Any] = options or {"do_remove_nan": True}
        self.naming_convention: str = naming_convention

        # Extract required keys from the naming convention
        self.special_keys = ["batch_num", "sample_num", "iter"]
        self.required_keys = [
            key
            for key in re.findall(r"\{(.*?)\}", self.naming_convention)
            if key not in self.special_keys
        ]

        self.file_format: str = file_format.lower()

        # Create output directory if it doesn't exist
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            f"Exporter initialized. Output directory: {self.output_directory}"
        )

        # Determine number of workers using get_max_workers utility
        self.max_workers = get_max_workers()
        self.logger.info(
            f"Initializing ThreadPoolExecutor with {self.max_workers} workers."
        )

        # Initialize ThreadPoolExecutor for parallel exporting
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=self.max_workers
        )
        self.logger.info("Exporter ThreadPoolExecutor initialized successfully.")

        # Initialize a lock and a mapping to keep track of the iterator per base filename
        self.iter_lock: threading.Lock = threading.Lock()
        self.iter_map: Dict[str, int] = defaultdict(int)

        # Initialize DataHandler
        self.data_handler: DataHandler = DataHandler(logger=self.logger)

        # Initialize the appropriate file writer
        if self.file_format == "fxye":
            self.file_writer = FxyeWriter(self.output_directory, self.logger)
            self.logger.info("FxyeWriter initialized.")
        elif self.file_format == "xye":
            self.file_writer = XyeWriter(self.output_directory, self.logger)
            self.logger.info("XyeWriter initialized.")
        elif self.file_format == "xy":
            self.file_writer = XyWriter(self.output_directory, self.logger)
            self.logger.info("XyWriter initialized.")
        elif self.file_format == "tiff":
            self.file_writer = TiffWriter(self.output_directory, self.logger)
            self.logger.info("TiffWriter initialized.")
        else:
            self.logger.error(f"Unsupported file format: {self.file_format}")
            raise ValueError(f"Unsupported file format: {self.file_format}")

    def process_sample(
        self,
        batch_num: int,
        sample_num: int,
        sample_data: np.ndarray,
        metadata: tf.Tensor,
    ) -> None:
        """
        Processes a single sample: decodes metadata, applies naming convention, validates data,
        and exports the data.

        Parameters
        ----------
        batch_num : int
            Current batch number.
        sample_num : int
            Current sample number within the batch.
        sample_data : np.ndarray
            Integrated or unintegrated diffractogram data.
        metadata : tf.Tensor
            Metadata tensor or object.
        """
        try:
            # Decode and parse metadata
            metadata_dict: Optional[Dict[str, Any]] = (
                self.data_handler.decode_and_parse_metadata(metadata.numpy())
            )
            if metadata_dict is None:
                self.logger.error(
                    f"Failed to decode and parse metadata for batch {batch_num}, sample {sample_num}"
                )
                return

            # Extract required naming variables
            naming_variables: Dict[str, Any] = (
                self.data_handler.extract_naming_variables(
                    metadata_dict, self.required_keys
                )
            )

            # Add batch and sample numbers
            naming_variables.update(
                {
                    "batch_num": batch_num,
                    "sample_num": sample_num,
                    "iter": 0,  # Placeholder, will set later
                }
            )

            # Generate base filename without 'iter' placeholder
            base_naming_convention: str = self.naming_convention.replace(
                "{iter}", ""
            ).replace("__", "_")
            base_filename: str = base_naming_convention.format(
                **naming_variables
            ).strip("_")

            # Update iterator safely
            with self.iter_lock:
                current_iter: int = self.iter_map[base_filename]
                naming_variables["iter"] = current_iter
                self.iter_map[base_filename] += 1

            # Generate full filename
            try:
                filename: str = self.naming_convention.format(**naming_variables)
            except Exception as exc:
                self.logger.error(f"Error applying naming convention: {exc}")
                filename = f"batch{batch_num}_sample{sample_num}_iter{current_iter}"

            # Validate data dimensions based on file_format
            if self.file_format in ["fxye", "xye", "xy"]:
                # Expected data shape is (npt, 2) or (npt, 3)
                if sample_data.ndim != 2 or sample_data.shape[1] < 2:
                    self.logger.error(
                        f"Data shape {sample_data.shape} not compatible with {self.file_format} format for batch {batch_num}, sample {sample_num}"
                    )
                    return

                x = sample_data[:, 0]
                y = sample_data[:, 1]
                if sample_data.shape[1] >= 3:
                    e = sample_data[:, 2]
                else:
                    e = np.zeros_like(y)

                # Ensure data is 1D
                if x.ndim != 1 or y.ndim != 1 or e.ndim != 1:
                    self.logger.error(
                        f"Data dimensions invalid for {self.file_format} format for batch {batch_num}, sample {sample_num}"
                    )
                    return

                # Remove NaNs if specified
                if self.options.get("do_remove_nan", True):
                    valid_indices: np.ndarray = (
                        ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(e)
                    )
                    if not np.any(valid_indices):
                        self.logger.warning(
                            f"All data contains NaNs for batch {batch_num}, sample {sample_num}. Skipping export."
                        )
                        return
                    x, y, e = x[valid_indices], y[valid_indices], e[valid_indices]

                # Export the data
                if self.file_format == "fxye":
                    self.file_writer.write(x, y, e, filename)
                elif self.file_format == "xye":
                    self.file_writer.write(x, y, e, filename)
                elif self.file_format == "xy":
                    self.file_writer.write(x, y, filename)

            elif self.file_format == "tiff":
                # Expected data is 2D array
                if sample_data.ndim != 2:
                    self.logger.error(
                        f"Data shape {sample_data.shape} not compatible with TIFF format for batch {batch_num}, sample {sample_num}"
                    )
                    return

                data = sample_data

                # Remove NaNs if specified
                if self.options.get("do_remove_nan", True):
                    data = np.nan_to_num(data)

                # Export the image
                self.file_writer.write(data, filename)

            else:
                self.logger.error(f"Unsupported file format: {self.file_format}")
                raise ValueError(f"Unsupported file format: {self.file_format}")

            self.logger.debug(f"Exported file saved: {filename}")

        except Exception as exc:
            self.logger.error(
                f"Error processing batch {batch_num}, sample {sample_num}: {exc}",
                exc_info=True,
            )

    def export_dataset(self, dataset: tf.data.Dataset) -> None:
        """
        Exports each batch in the dataset to files in the specified format with data integrity protections.

        Parameters
        ----------
        dataset : tf.data.Dataset
            Dataset containing batches of diffractogram data and their metadata.
        """
        self.logger.info("Starting dataset export.")

        try:
            for batch_num, (data_batch, metadata_batch) in enumerate(dataset):
                batch_size: int = data_batch.shape[0]
                self.logger.debug(
                    f"Exporting batch {batch_num} with {batch_size} samples."
                )

                futures = []
                for sample_num in range(batch_size):
                    sample_data: np.ndarray = data_batch[sample_num].numpy()
                    metadata: tf.Tensor = metadata_batch[sample_num]
                    # Submit the sample processing to the executor
                    future = self.executor.submit(
                        self.process_sample,
                        batch_num,
                        sample_num,
                        sample_data,
                        metadata,
                    )
                    futures.append(future)

                # Wait for all futures to complete and handle exceptions
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        self.logger.error(
                            f"Exporting sample resulted in an exception: {exc}",
                            exc_info=True,
                        )

        except Exception as exc:
            self.logger.error(f"Error during dataset export: {exc}", exc_info=True)

        finally:
            self.executor.shutdown(wait=True)
            self.logger.info("Dataset export completed.")

            # Clear TensorFlow session and perform garbage collection
            tf.keras.backend.clear_session()
            gc.collect()
            self.logger.info("TensorFlow session cleared and garbage collected.")

    def shutdown(self):
        """Clean up resources used by the Exporter."""
        try:
            if hasattr(self, "executor"):
                self.executor.shutdown(wait=True)
                self.logger.info("Exporter ThreadPoolExecutor shutdown completed.")
            tf.keras.backend.clear_session()
            gc.collect()
            self.logger.info("TensorFlow session cleared and garbage collected.")
        except Exception as e:
            self.logger.error(f"Error during exporter shutdown: {e}", exc_info=True)
