# FAIRshake/integration/integration.py

import gc
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

from FAIRshake.utils.resource_utils import get_max_workers


class Integrator:
    """
    A class for integrating diffractogram images using PyFAI's AzimuthalIntegrator.
    """

    def __init__(
        self,
        integration_config: Dict[str, Any],
    ):
        """
        Initialize the Integrator.

        Parameters
        ----------
        integration_config : Dict[str, Any]
            Configuration parameters for integration, including 'poni_file_path'.
        """
        self.logger: logging.Logger = logging.getLogger(
            "fairshake.integration.Integrator"
        )
        self.logger.setLevel(logging.INFO)

        # Retrieve 'poni_file_path' from integration_config
        poni_file_path = integration_config.get("poni_file_path")
        if not poni_file_path:
            self.logger.error(
                "Integration configuration must include 'poni_file_path'."
            )
            raise ValueError("Integration configuration must include 'poni_file_path'.")

        # Ensure that 'poni_file_path' is an absolute path
        poni_file_path = os.path.abspath(poni_file_path)
        if not os.path.exists(poni_file_path):
            self.logger.error(f"PONI file {poni_file_path} not found.")
            raise FileNotFoundError(f"PONI file {poni_file_path} not found.")

        self.logger.info(
            f"Loading azimuthal integrator with PONI file: {poni_file_path}"
        )
        self.integrator: AzimuthalIntegrator = AzimuthalIntegrator()
        self.integrator.load(poni_file_path)

        # Store configuration for integration
        self.integration_config: Dict[str, Any] = integration_config

        # Determine number of workers using get_max_workers utility
        self.max_workers = get_max_workers()
        self.logger.info(
            f"Initializing ThreadPoolExecutor with {self.max_workers} workers."
        )

        # Initialize ThreadPoolExecutor
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=self.max_workers
        )

        self.logger.info("Integrator initialized successfully.")

    def integrate_single_image(self, image: np.ndarray) -> np.ndarray:
        """
        Performs azimuthal integration on a single image.

        Parameters
        ----------
        image : np.ndarray
            2D array representing the image.

        Returns
        -------
        np.ndarray
            Integrated result with shape (npt, 2) or (npt, 3) containing [radial, intensity] or [radial, intensity, sigma].
        """
        if image.ndim != 2:
            self.logger.error(f"Image must be 2D; got shape {image.shape}")
            raise ValueError(f"Image must be 2D; got shape {image.shape}")

        try:
            # Perform azimuthal integration
            # Check if error_model is specified
            error_model = self.integration_config.get("error_model", None)
            integration_method = self.integration_config.get("method", "full")
            npt_radial = self.integration_config.get("npt_radial", 1000)
            unit = self.integration_config.get("unit", "2th_deg")
            safe = self.integration_config.get("safe", True)
            polarization_factor = self.integration_config.get(
                "polarization_factor", None
            )
            dark = self.integration_config.get("dark", None)
            flat = self.integration_config.get("flat", None)
            mask = self.integration_config.get("mask", None)
            dummy = self.integration_config.get("dummy", None)
            delta_dummy = self.integration_config.get("delta_dummy", None)
            normalization_factor = self.integration_config.get(
                "normalization_factor", 1.0
            )
            radial_range = self.integration_config.get("radial_range", None)
            azimuth_range = self.integration_config.get("azimuth_range", None)

            if error_model is not None:
                result = self.integrator.integrate1d(
                    image,
                    npt=npt_radial,
                    unit=unit,
                    error_model=error_model,
                    method=integration_method,
                    safe=safe,
                    polarization_factor=polarization_factor,
                    dark=dark,
                    flat=flat,
                    mask=mask,
                    dummy=dummy,
                    delta_dummy=delta_dummy,
                    normalization_factor=normalization_factor,
                    radial_range=radial_range,
                    azimuth_range=azimuth_range,
                )
                # Stack and transpose to (npt, 3)
                integrated_result: np.ndarray = np.stack(
                    [result.radial, result.intensity, result.sigma], axis=1
                )
            else:
                result = self.integrator.integrate1d(
                    image,
                    npt=npt_radial,
                    unit=unit,
                    method=integration_method,
                    safe=safe,
                    polarization_factor=polarization_factor,
                    dark=dark,
                    flat=flat,
                    mask=mask,
                    dummy=dummy,
                    delta_dummy=delta_dummy,
                    normalization_factor=normalization_factor,
                    radial_range=radial_range,
                    azimuth_range=azimuth_range,
                )
                # Stack and transpose to (npt, 2)
                integrated_result: np.ndarray = np.stack(
                    [result.radial, result.intensity], axis=1
                )
            self.logger.debug(
                f"Integrated result shape confirmed: {integrated_result.shape}"
            )

            return integrated_result

        except Exception as e:
            self.logger.error(
                f"Error during integration of single image: {e}", exc_info=True
            )
            raise

    def integrate_map_fn(
        self, image_batch: tf.Tensor, metadata_batch: tf.Tensor
    ) -> tf.Tensor:
        """
        Function to handle integration for a batch of images.

        Parameters
        ----------
        image_batch : tf.Tensor
            Batch of images with shape (batch_size, height, width).
        metadata_batch : tf.Tensor
            Corresponding metadata for each image in the batch.

        Returns
        -------
        tf.Tensor
            Integrated results for each image in the batch with shape (batch_size, npt, 2 or 3).
        tf.Tensor
            Corresponding metadata for each image in the batch.
        """

        def np_integrate(images: np.ndarray) -> np.ndarray:
            """
            Integrates a batch of images using ThreadPoolExecutor for parallel processing.

            Parameters
            ----------
            images : np.ndarray
                Array of images with shape (batch_size, height, width).

            Returns
            -------
            np.ndarray
                Integrated results with shape (batch_size, npt, 2 or 3).
            """
            batch_size: int = images.shape[0]
            self.logger.debug(f"Starting integration for batch of size: {batch_size}")

            integrated_results: List[np.ndarray] = [None] * batch_size
            futures = {
                self.executor.submit(self.integrate_single_image, images[i]): i
                for i in range(batch_size)
            }

            for future in as_completed(futures):
                i: int = futures[future]
                try:
                    result: np.ndarray = future.result()
                    integrated_results[i] = result
                except Exception as exc:
                    self.logger.error(
                        f"Integration failed for image {i}: {exc}", exc_info=True
                    )
                    # Append NaNs or handle as needed
                    npt: int = self.integration_config.get("npt_radial", 1000)
                    if self.integration_config.get("error_model", None) is not None:
                        integrated_results[i] = np.full(
                            (npt, 3), np.nan, dtype=np.float32
                        )
                    else:
                        integrated_results[i] = np.full(
                            (npt, 2), np.nan, dtype=np.float32
                        )

            # Convert list to numpy array
            integrated_array: np.ndarray = np.array(
                integrated_results,
                dtype=np.float32,
            )
            self.logger.debug(
                f"Batch integration completed. Integrated array shape: {integrated_array.shape}"
            )
            return integrated_array

        # Use tf.numpy_function to process the batch efficiently
        integrated_tensor: tf.Tensor = tf.numpy_function(
            np_integrate, [image_batch], tf.float32
        )
        # Set the shape information (optional but recommended)
        npt = self.integration_config.get("npt_radial", 1000)
        if self.integration_config.get("error_model", None) is not None:
            integrated_tensor.set_shape((None, npt, 3))  # None for dynamic batch size
        else:
            integrated_tensor.set_shape((None, npt, 2))  # None for dynamic batch size

        return integrated_tensor, metadata_batch

    def integrate_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Method to integrate an entire dataset, applying integration to each image in the batch.

        Parameters
        ----------
        dataset : tf.data.Dataset
            Dataset containing batches of images and their metadata.

        Returns
        -------
        tf.data.Dataset
            Dataset with integration applied to each image in each batch.
        """

        integrated_dataset: tf.data.Dataset = dataset.map(
            self.integrate_map_fn,
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)

        return integrated_dataset

    def shutdown(self) -> None:
        """
        Shuts down the ThreadPoolExecutor and clears TensorFlow sessions.
        """
        try:
            self.logger.info("Shutting down ThreadPoolExecutor.")
            self.executor.shutdown(wait=True)
            self.logger.info("ThreadPoolExecutor shutdown completed.")
        except Exception as e:
            self.logger.error(f"Error during executor shutdown: {e}", exc_info=True)

        try:
            self.logger.info("Clearing TensorFlow session.")
            tf.keras.backend.clear_session()
            gc.collect()
            self.logger.info("TensorFlow session cleared and garbage collected.")
        except Exception as e:
            self.logger.error(
                f"Error during TensorFlow session clearing: {e}", exc_info=True
            )
