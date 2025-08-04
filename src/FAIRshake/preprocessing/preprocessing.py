# FAIRshake/preprocessing/preprocessing.py

import logging
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import tensorflow as tf

from FAIRshake.data_loading.file_loader import FileLoader


class Preprocessor:
    """
    Preprocessor class for performing dark correction, masking, and intensity clipping.
    """

    def __init__(
        self,
        dark_field_path: Optional[Union[str, Path]] = None,
        base_path: Optional[Union[str, Path]] = None,
        mask_file_path: Optional[Union[str, Path]] = None,
        invert_mask: bool = False,
        min_intensity: Optional[float] = None,
        max_intensity: Optional[float] = None,
    ):
        """
        Initialize the Preprocessor.

        Parameters
        ----------
        dark_field_path : str or Path, optional
            Relative path to the dark field file.
        base_path : str or Path, optional
            Base path to resolve relative paths.
        mask_file_path : str or Path, optional
            Relative path to the mask file.
        invert_mask : bool, optional
            Whether to invert the mask values during loading. Default is False.
        min_intensity : float, optional
            Minimum intensity value for clipping.
        max_intensity : float, optional
            Maximum intensity value for clipping.
        """
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.base_path: Optional[Path] = Path(base_path) if base_path else None
        self.file_loader: FileLoader = FileLoader(
            log_dir=None
        )  # Assuming no logging in FileLoader here

        # Load and cache the dark field
        self.dark_field: Optional[tf.Tensor] = None
        if dark_field_path:
            self.dark_field = self._load_and_cache_dark_field(dark_field_path)
            if self.dark_field is not None:
                self.logger.info(
                    f"Dark field loaded and cached from '{dark_field_path}'"
                )
            else:
                self.logger.warning(
                    f"Failed to load dark field from '{dark_field_path}'"
                )

        # Load and cache the mask
        self.mask: Optional[tf.Tensor] = None
        if mask_file_path:
            self.mask = self._load_and_cache_mask(
                mask_file_path, invert_mask=invert_mask
            )
            if self.mask is not None:
                self.logger.info(
                    f"Mask file loaded and cached from '{mask_file_path}' (invert_mask={invert_mask})"
                )
            else:
                self.logger.warning(f"Failed to load mask file from '{mask_file_path}'")

        # Set intensity clipping values
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

        # Determine the smallest positive value based on the image dtype (assuming float32)
        self.tiny_value = tf.constant(
            np.finfo(np.float32).tiny, dtype=tf.float32
        )  # 1.17549435e-38 for float32

    def _load_and_cache_dark_field(
        self, relative_path: Union[str, Path]
    ) -> Optional[tf.Tensor]:
        """
        Load the dark field file, average it, and cache the result.

        Parameters
        ----------
        relative_path : str or Path
            Relative path to the dark field file.

        Returns
        -------
        tf.Tensor or None
            TensorFlow tensor of the averaged dark field, or None if loading fails.
        """
        file_path: Path = (
            self.base_path / relative_path if self.base_path else Path(relative_path)
        )

        # Check if the dark field is already cached
        cache_path: Path = file_path.with_suffix(".npy")
        if cache_path.exists():
            self.logger.info(f"Loading cached dark field from '{cache_path}'")
            try:
                data: np.ndarray = np.load(cache_path)
            except Exception as e:
                self.logger.error(
                    f"Error loading cached dark field from '{cache_path}': {e}"
                )
                return None
        else:
            data, _ = self.file_loader.load_file(
                file_path, average_frames=True, invert_mask=False
            )
            if data is not None:
                try:
                    # Save the averaged dark field to cache
                    np.save(cache_path, data)
                    self.logger.info(f"Dark field cached at '{cache_path}'")
                except Exception as e:
                    self.logger.error(
                        f"Error caching dark field to '{cache_path}': {e}"
                    )
            else:
                self.logger.error(f"Failed to load dark field from '{file_path}'")
                return None

        # Convert to float32 Tensor
        dark_field_tensor: tf.Tensor = tf.convert_to_tensor(data, dtype=tf.float32)
        return dark_field_tensor

    def _load_and_cache_mask(
        self, mask_file_path: Union[str, Path], invert_mask: bool
    ) -> Optional[tf.Tensor]:
        """
        Load the mask file, optionally invert it, and cache the result.

        Parameters
        ----------
        mask_file_path : str or Path
            Path to the mask file.
        invert_mask : bool
            Whether to invert the mask values during loading.

        Returns
        -------
        tf.Tensor or None
            TensorFlow tensor of the mask, or None if loading fails.
        """
        file_path: Path = (
            self.base_path / mask_file_path if self.base_path else Path(mask_file_path)
        )

        # Define a unique cache filename based on the inversion flag
        cache_suffix = "_inverted" if invert_mask else "_original"
        cache_path: Path = file_path.with_suffix(f".npy{cache_suffix}")

        if cache_path.exists():
            self.logger.info(f"Loading cached mask from '{cache_path}'")
            try:
                data: np.ndarray = np.load(cache_path)
            except Exception as e:
                self.logger.error(f"Error loading cached mask from '{cache_path}': {e}")
                return None
        else:
            data, metadata = self.file_loader.load_file(
                file_path,
                average_frames=False,
                return_frames=False,
                invert_mask=False,  # Handle inversion manually
            )
            if data is None:
                self.logger.error(f"Failed to load mask data from '{file_path}'.")
                return None

            # If multiple frames, take the first one or handle accordingly
            if isinstance(data, list):
                self.logger.warning(
                    f"Multiple frames detected in mask file '{file_path}'. Using the first frame."
                )
                data = data[0] if data else None
                if data is None:
                    self.logger.error(f"No frames found in mask file '{file_path}'.")
                    return None

            # Ensure mask is binary (1 for valid pixels, 0 for invalid)
            mask_array: np.ndarray = (np.array(data) > 0).astype(np.float32)

            if invert_mask:
                mask_array = 1.0 - mask_array  # Invert the mask

            # Save the processed mask to cache
            try:
                np.save(cache_path, mask_array)
                self.logger.info(
                    f"Mask cached at '{cache_path}' (invert_mask={invert_mask})"
                )
            except Exception as e:
                self.logger.error(f"Error caching mask to '{cache_path}': {e}")
                return None

            data = mask_array  # Update data with the processed mask

        # Convert to float32 Tensor
        mask_tensor: tf.Tensor = tf.convert_to_tensor(data, dtype=tf.float32)

        # Log mask statistics for debugging
        try:
            mask_min = tf.reduce_min(mask_tensor).numpy()
            mask_max = tf.reduce_max(mask_tensor).numpy()
            mask_unique = tf.unique(tf.reshape(mask_tensor, [-1]))[0].numpy()
            self.logger.info(
                f"Mask statistics - min: {mask_min}, max: {mask_max}, unique values: {mask_unique}"
            )
        except Exception as e:
            self.logger.warning(f"Could not compute mask statistics: {e}")

        return mask_tensor

    def preprocess_image(self, image: tf.Tensor) -> tf.Tensor:
        """
        Apply dark correction, masking, and intensity clipping to the input image.

        Parameters
        ----------
        image : tf.Tensor
            Input image tensor.

        Returns
        -------
        tf.Tensor
            Preprocessed image tensor.
        """
        # Apply dark correction
        if self.dark_field is not None:
            image = tf.subtract(image, self.dark_field)
            self.logger.debug("Applied dark field correction.")
        else:
            self.logger.warning("No dark field loaded; skipping dark correction.")

        # Apply mask
        if self.mask is not None:
            # Verify that mask and image have the same dimensions
            # Correctly extract height and width by excluding the batch dimension
            image_shape = image.shape.as_list()[1:3]  # [height, width]
            mask_shape = self.mask.shape.as_list()  # [height, width]
            self.logger.debug(f"Image shape: {image_shape}, Mask shape: {mask_shape}")

            if image_shape != mask_shape:
                self.logger.error(
                    f"Mask dimensions {mask_shape} do not match image dimensions {image_shape}."
                )
                raise ValueError("Mask dimensions do not match image dimensions.")

            # Apply mask: set masked pixels to tiny_value instead of zero
            image = tf.where(self.mask > 0, image, self.tiny_value)
            self.logger.debug(
                "Applied mask to the image, setting masked pixels to tiny value."
            )
        else:
            self.logger.warning("No mask loaded; skipping masking.")

        # Clip intensity values
        if self.min_intensity is not None or self.max_intensity is not None:
            clip_min = (
                self.min_intensity
                if self.min_intensity is not None
                else tf.reduce_min(image)
            )
            clip_max = (
                self.max_intensity
                if self.max_intensity is not None
                else tf.reduce_max(image)
            )
            image = tf.clip_by_value(
                image,
                clip_value_min=clip_min,
                clip_value_max=clip_max,
            )
            self.logger.debug(
                f"Clipped intensity values to min: {clip_min}, max: {clip_max}"
            )

        return image

    def preprocess_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Apply preprocessing to a dataset.

        Parameters
        ----------
        dataset : tf.data.Dataset
            Dataset containing images to be preprocessed.

        Returns
        -------
        tf.data.Dataset
            Preprocessed dataset.
        """

        def preprocess(
            image: tf.Tensor, metadata: tf.Tensor
        ) -> Tuple[tf.Tensor, tf.Tensor]:
            preprocessed_image = self.preprocess_image(image)
            return preprocessed_image, metadata

        return dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
