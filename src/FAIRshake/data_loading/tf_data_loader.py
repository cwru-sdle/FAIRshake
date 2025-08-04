# FAIRshake/data_loading/tf_data_loader.py

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from .file_loader import FileLoader
from FAIRshake.data_handling.data_handler import DataHandler


class TfDataLoader:
    """
    TensorFlow data loader for FAIRshake using tf.data.Dataset.
    """

    def __init__(
        self,
        base_path: Union[str, Path],
        require_metadata: bool = True,
        load_metadata_files: bool = True,
        load_detector_metadata: bool = False,
        data_file_types: Optional[List[str]] = None,
        metadata_file_types: Optional[List[str]] = None,
        require_all_formats: bool = False,
        average_frames: bool = False,
        log_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the TfDataLoader with configuration parameters.

        Parameters
        ----------
        base_path : str or Path
            Base directory path where data files are located.
        require_metadata : bool, optional
            Whether to require metadata files for image files.
            Default is True.
        load_metadata_files : bool, optional
            Whether to load additional metadata files.
            Default is True.
        load_detector_metadata : bool, optional
            Whether to load detector-specific metadata.
            Default is False.
        data_file_types : list of str, optional
            List of data file extensions to include. If None, default extensions are used.
            Default is None.
        metadata_file_types : list of str, optional
            List of metadata file extensions to include. If None, default extensions are used.
            Default is None.
        require_all_formats : bool, optional
            Whether to require all metadata file types to be present.
            Default is False.
        average_frames : bool, optional
            Whether to average multiple frames into a single frame.
            Default is False.
        log_dir : str or Path, optional
            Directory for storing logs. If None, logging is configured to output
            to console.
            Default is None.
        """
        self.base_path: Path = Path(base_path)
        self.require_metadata: bool = require_metadata
        self.load_metadata_files: bool = load_metadata_files
        self.load_detector_metadata: bool = load_detector_metadata
        self.data_file_types: List[str] = data_file_types or [
            ".ge2",
            ".tif",
            ".edf",
            ".cbf",
            ".mar3450",
            ".h5",
            ".png",
        ]
        self.metadata_file_types: List[str] = metadata_file_types or [
            ".json",
            ".poni",
            ".instprm",
            ".geom",
            ".spline",
        ]
        self.require_all_formats: bool = require_all_formats
        self.average_frames: bool = average_frames
        self.log_dir: Optional[Path] = Path(log_dir) if log_dir else None

        # Initialize logger without adding separate handlers
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # Set to INFO or WARNING as needed

        self.logger.info("Initializing TfDataLoader.")

        self.file_loader: FileLoader = FileLoader(log_dir=self.log_dir)
        self.data_handler: DataHandler = DataHandler(logger=self.logger)

    def create_dataset(
        self, pattern: str = "**/*", batch_size: int = 32
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset that loads each frame from image files.

        Parameters
        ----------
        pattern : str, optional
            Glob pattern to match files. Default is "**/*".
        batch_size : int, optional
            Number of samples per batch. Default is 32.

        Returns
        -------
        tf.data.Dataset
            A TensorFlow dataset object.
        """
        self.logger.info("Creating TensorFlow dataset using tf.data API.")

        image_files: List[Path] = self._gather_image_files(pattern)
        total_images_found: int = len(image_files)
        total_metadata_files_found: int = self._count_metadata_files()

        self.logger.info(f"Total image files found: {total_images_found}")
        self.logger.info(f"Total metadata files found: {total_metadata_files_found}")

        if total_images_found == 0:
            self.logger.error(
                "No image files found after filtering. Exiting dataset creation."
            )
            return tf.data.Dataset.from_tensors(
                (
                    tf.constant([], dtype=tf.float32),
                    tf.constant("", dtype=tf.string),
                )
            )

        # Create a generator function that yields data and metadata for each frame
        def data_generator() -> Generator[Tuple[np.ndarray, str], None, None]:
            for file_path in image_files:
                data_list, metadata_list = self._load_data(str(file_path))
                if data_list is not None and metadata_list is not None:
                    for data, metadata_json in zip(data_list, metadata_list):
                        if data.size > 0:
                            yield data, metadata_json
                        else:
                            # For metadata-only entries, yield empty array and metadata
                            yield np.array([], dtype=np.float32), metadata_json
                else:
                    self.logger.warning(f"No data loaded from file '{file_path}'.")

        # Create the dataset from the generator
        dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.string),
            ),
        )

        # Batch and prefetch the dataset
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        self.logger.info("TensorFlow dataset created successfully.")
        return dataset

    def _load_data(
        self, file_path_str: str
    ) -> Tuple[Optional[List[np.ndarray]], Optional[List[str]]]:
        """
        Load image data and sanitized metadata for a single file.

        Parameters
        ----------
        file_path_str : str
            Path to the file to load.

        Returns
        -------
        tuple
            A tuple of (data_list, metadata_json_list). Each list contains the
            data arrays and their corresponding metadata in JSON format.
            Returns (None, None) if loading fails.
        """
        data, metadata = self.file_loader.load_file(
            file_path_str,
            average_frames=self.average_frames,
            return_frames=True,
        )

        if data is None and metadata is None:
            self.logger.warning(
                f"Data and metadata not loaded for '{file_path_str}'. Skipping."
            )
            return None, None

        data_list: List[np.ndarray] = []
        metadata_json_list: List[str] = []

        if isinstance(data, list):
            # Handle list of frames/data
            for i, frame_data in enumerate(data):
                frame_metadata = metadata[i] if metadata and i < len(metadata) else {}
                data_array, metadata_json = self._process_frame_data_and_metadata(
                    frame_data, frame_metadata, file_path_str
                )
                data_list.append(data_array)
                metadata_json_list.append(metadata_json)
        elif isinstance(data, np.ndarray):
            # Handle single frame data
            frame_metadata = metadata if metadata else {}
            data_array, metadata_json = self._process_frame_data_and_metadata(
                data, frame_metadata, file_path_str
            )
            data_list.append(data_array)
            metadata_json_list.append(metadata_json)
        elif isinstance(data, dict):
            # Handle dictionary data as metadata
            sanitized_metadata = self.data_handler.sanitize_metadata_keys(data)
            metadata_json = json.dumps(sanitized_metadata)
            # For metadata-only entries, data is an empty array
            data_list.append(np.array([], dtype=np.float32))
            metadata_json_list.append(metadata_json)
        else:
            self.logger.error(
                f"Unexpected data type returned from file loader for '{file_path_str}': {type(data)}"
            )
            return None, None

        return data_list, metadata_json_list

    def _process_frame_data_and_metadata(
        self, frame_data: Any, frame_metadata: Dict[str, Any], file_path_str: str
    ) -> Tuple[np.ndarray, str]:
        """
        Process individual frame data and metadata.

        Parameters
        ----------
        frame_data : Any
            The data for the frame, expected to be a NumPy array or similar.
        frame_metadata : Dict[str, Any]
            Metadata associated with the frame.
        file_path_str : str
            Path to the file being processed.

        Returns
        -------
        tuple
            A tuple of (data_array, metadata_json).
        """
        # If frame_data is bytes or not numeric, convert as appropriate
        if isinstance(frame_data, bytes):
            # For bytes data, attempt to decode or skip
            try:
                # Attempt to interpret bytes as string and store as metadata
                frame_data_decoded = frame_data.decode("utf-8", "replace")
                self.logger.warning(
                    f"Bytes data encountered in '{file_path_str}'. Skipping conversion to array."
                )
                data_array = np.array([], dtype=np.float32)
            except Exception as e:
                self.logger.error(
                    f"Cannot decode bytes data for '{file_path_str}': {e}"
                )
                data_array = np.array([], dtype=np.float32)
        elif isinstance(frame_data, (list, tuple)):
            # If frame_data is a list or tuple, convert to numpy array
            data_array = np.array(frame_data, dtype=np.float32)
        elif isinstance(frame_data, np.ndarray):
            if frame_data.dtype.kind not in ["u", "i", "f"]:  # Not a numeric type
                self.logger.warning(
                    f"Non-numeric array data encountered for '{file_path_str}'. Skipping conversion."
                )
                data_array = np.array([], dtype=np.float32)
            else:
                # Ensure correct dtype
                data_array = frame_data.astype(np.float32)
        elif isinstance(frame_data, dict):
            # Handle dictionary data (e.g., integrator metadata)
            sanitized_metadata = self.data_handler.sanitize_metadata_keys(frame_data)
            metadata_json = json.dumps(sanitized_metadata)
            data_array = np.array([], dtype=np.float32)
            return data_array, metadata_json
        else:
            # If it's neither numeric nor dictionary, treat as empty data
            self.logger.warning(
                f"Unsupported data type {type(frame_data)} for '{file_path_str}'."
            )
            data_array = np.array([], dtype=np.float32)

        # Combine frame_metadata if necessary
        combined_metadata = self._combine_metadata(file_path_str, frame_metadata)
        sanitized_metadata = self.data_handler.sanitize_metadata_keys(combined_metadata)
        metadata_json = json.dumps(sanitized_metadata)
        return data_array, metadata_json

    def _gather_image_files(self, pattern: str) -> List[Path]:
        """
        Gather image files based on the pattern and filter them based on metadata presence.

        Parameters
        ----------
        pattern : str
            Glob pattern to match files.

        Returns
        -------
        list of Path
            List of image file paths.
        """
        image_files: List[Path] = []
        for ext in self.data_file_types:
            found: List[Path] = list(self.base_path.rglob(f"{pattern}{ext}"))
            self.logger.debug(f"Found {len(found)} image files with extension '{ext}'.")
            for img_path in found:
                if self.require_metadata and not self._has_metadata(img_path):
                    self.logger.debug(f"Excluding {img_path} due to missing metadata.")
                    continue
                if self.require_all_formats and not self._has_all_metadata_files(
                    img_path
                ):
                    self.logger.debug(
                        f"Excluding {img_path} due to incomplete metadata formats."
                    )
                    continue
                image_files.append(img_path)
        return image_files

    def _count_metadata_files(self) -> int:
        """
        Count all metadata files based on their extensions.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Total number of metadata files found.
        """
        total_metadata_files: int = sum(
            len(list(self.base_path.rglob(f"**/*{ext}")))
            for ext in self.metadata_file_types
        )
        return total_metadata_files

    def _has_metadata(self, img_path: Path) -> bool:
        """
        Check if the image file has at least one metadata file.

        Parameters
        ----------
        img_path : Path
            Path to the image file.

        Returns
        -------
        bool
            True if at least one metadata file is present, False otherwise.
        """
        img_dir: Path = img_path.parent
        for ext in self.metadata_file_types:
            if any(img_dir.glob(f"*{ext}")):
                return True
        return False

    def _has_all_metadata_files(self, img_path: Path) -> bool:
        """
        Check if all required metadata file types are present in the image's directory.

        Parameters
        ----------
        img_path : Path
            Path to the image file.

        Returns
        -------
        bool
            True if all metadata file types are present, False otherwise.
        """
        img_dir: Path = img_path.parent
        return all(any(img_dir.glob(f"*{ext}")) for ext in self.metadata_file_types)

    def _combine_metadata(
        self, img_path: str, frame_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine frame-specific metadata and additional metadata based on flags.

        Parameters
        ----------
        img_path : str
            Path to the image file.
        frame_metadata : dict
            Metadata associated with the frame.

        Returns
        -------
        dict
            Combined metadata dictionary.
        """
        combined: Dict[str, Any] = {}

        # Include detector metadata if flag is set
        if self.load_detector_metadata and frame_metadata:
            filtered_metadata: Dict[str, Any] = self._filter_metadata(frame_metadata)
            if filtered_metadata:
                combined["detector_metadata"] = (
                    self.data_handler.decode_and_parse_metadata(filtered_metadata)
                )
                self.logger.debug(f"Including 'detector_metadata' for '{img_path}'.")

        # Load additional metadata from metadata files if flag is set
        if self.load_metadata_files:
            additional: Dict[str, Any] = self._load_additional_metadata(img_path)
            combined.update(additional)

        return combined

    def _filter_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter out detector-specific metadata based on the flag.

        Parameters
        ----------
        metadata : dict
            Metadata dictionary.

        Returns
        -------
        dict
            Filtered metadata dictionary.
        """
        # Implement any filtering logic if needed
        return metadata

    def _load_additional_metadata(self, img_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load all additional metadata files in the image's directory.

        Parameters
        ----------
        img_path : str or Path
            Path to the image file.

        Returns
        -------
        dict
            Dictionary containing all additional metadata.
        """
        metadata: Dict[str, Any] = {}
        img_dir: Path = Path(img_path).parent

        for ext in self.metadata_file_types:
            for metadata_file in img_dir.glob(f"*{ext}"):
                data, _ = self.file_loader.load_file(
                    metadata_file,
                    average_frames=False,
                    return_frames=False,
                )
                if data:
                    key: str = f"metadata_{metadata_file.stem}"
                    decoded_data: Optional[Dict[str, Any]] = (
                        self.data_handler.decode_and_parse_metadata(data)
                    )
                    sanitized_key: str = key.replace(" ", "_")  # Sanitize the key
                    sanitized_data: Any = self.data_handler.sanitize_metadata_keys(
                        decoded_data
                    )
                    metadata[sanitized_key] = sanitized_data
                    self.logger.debug(
                        f"Loaded '{sanitized_key}' from '{metadata_file}'."
                    )
                else:
                    self.logger.warning(
                        f"Failed to load metadata file: '{metadata_file}'"
                    )
        return metadata
