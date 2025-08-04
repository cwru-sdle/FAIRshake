# FAIRshake/data_loading/file_loader.py

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union, List

import fabio
import h5py
import numpy as np
import pyFAI  # Ensure pyFAI is installed if you're using PONI files
from PIL import Image

from FAIRshake.utils.logger import setup_logging


class FileLoader:
    """
    Class responsible for loading different types of files.
    """

    def __init__(
        self, log_dir: Optional[Union[str, Path]] = None, load_metadata: bool = True
    ):
        """
        Initialize the FileLoader with optional logging directory and metadata
        loading preference.

        Parameters
        ----------
        log_dir : str or Path, optional
            Directory for storing logs. If None, logging is configured to output
            to console.
        load_metadata : bool, optional
            Whether to load metadata associated with files. Default is True.
        """
        self.logger, self.log_file, self.log_records = setup_logging(
            log_dir, logging.WARNING
        )
        self.load_metadata = load_metadata
        self.registered_loaders: Dict[str, Callable[..., Tuple[Any, Optional[Any]]]] = (
            {}
        )
        self._register_default_loaders()

    def _register_default_loaders(self) -> None:
        """Register default file loaders for known file types."""
        # Existing loaders
        self.register_loader(".ge2", self.load_ge2_file)
        self.register_loader(".tif", self.load_tiff_file)
        self.register_loader(".tiff", self.load_tiff_file)
        self.register_loader(".edf", self.load_edf_file)
        self.register_loader(".poni", self.load_poni_file)
        self.register_loader(".instprm", self.load_instprm_file)
        self.register_loader(".json", self.load_json_file)

        # New loaders
        self.register_loader(".h5", self.load_h5_file)
        self.register_loader(".cbf", self.load_cbf_file)
        self.register_loader(".mar3450", self.load_mar3450_file)
        self.register_loader(".geom", self.load_geom_file)
        self.register_loader(".spline", self.load_spline_file)
        self.register_loader(".png", self.load_png_file)

    def register_loader(
        self, extension: str, loader_function: Callable[..., Tuple[Any, Optional[Any]]]
    ) -> None:
        """
        Register a loader function for a specific file extension.

        Parameters
        ----------
        extension : str
            File extension (e.g., '.ge2').
        loader_function : callable
            Function to load files with the given extension.
        """
        self.registered_loaders[extension.lower()] = loader_function
        self.logger.debug(f"Registered loader for '{extension}' files.")

    def load_file(
        self,
        file_path: Union[str, Path],
        average_frames: bool = False,
        return_frames: bool = False,
        invert_mask: bool = False,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load a file using the appropriate loader based on its extension.

        Parameters
        ----------
        file_path : str or Path
            Path to the file to be loaded.
        average_frames : bool, optional
            Whether to average frames if the file contains multiple frames.
            Default is False.
        return_frames : bool, optional
            If True, return a list of frames and metadata instead of a single
            data array. Default is False.
        invert_mask : bool, optional
            Whether to invert the mask values (1s to 0s and 0s to 1s). Use this
            when loading mask files. Default is False.

        Returns
        -------
        tuple
            If return_frames is False, returns a tuple (data, metadata).
            If return_frames is True, returns a tuple of (data_list, metadata_list).
            Returns (None, None) if loading fails.
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        loader_function = self.registered_loaders.get(extension)
        if loader_function is None:
            self.logger.error(f"No loader registered for extension '{extension}'.")
            return None, None

        return loader_function(
            file_path,
            average_frames=average_frames,
            return_frames=return_frames,
            invert_mask=invert_mask,
        )

    # Existing loader methods
    def load_ge2_file(
        self,
        file_path: Path,
        average_frames: bool = False,
        return_frames: bool = False,
        invert_mask: bool = False,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load data from a GE2 file using fabio.

        Parameters
        ----------
        file_path : Path
            Path to the GE2 file.
        average_frames : bool, optional
            Whether to average frames if the file contains multiple frames.
            Default is False.
        return_frames : bool, optional
            If True, return a list of frames and metadata instead of a single
            data array. Default is False.
        invert_mask : bool, optional
            Whether to invert the mask values. Not applicable here.
            Default is False.

        Returns
        -------
        tuple
            Data and metadata as described above.
        """
        try:
            with fabio.open(str(file_path)) as series:
                n_frames = series.nframes
                data_list = []
                metadata_list = []

                for i in range(n_frames):
                    frame = series.getframe(i)
                    frame_data = frame.data.astype(np.float32)
                    data_list.append(frame_data)

                    frame_metadata = frame.header if self.load_metadata else {}
                    metadata_list.append(frame_metadata)

                if average_frames:
                    averaged_data = np.mean(data_list, axis=0)
                    self.logger.debug(f"Averaged {n_frames} frames from '{file_path}'.")
                    return averaged_data, metadata_list[:1]

                if return_frames:
                    self.logger.debug(f"Loaded {n_frames} frames from '{file_path}'.")
                    return data_list, metadata_list

                if n_frames > 0:
                    self.logger.debug(f"Loaded first frame from '{file_path}'.")
                    return data_list[0], metadata_list[0]
                else:
                    self.logger.warning(f"No frames found in '{file_path}'.")
                    return None, None

        except Exception as e:
            self.logger.error(f"Error loading GE2 file '{file_path}': {e}")
            return None, None

    def load_tiff_file(
        self,
        file_path: Path,
        average_frames: bool = False,
        return_frames: bool = False,
        invert_mask: bool = False,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load data from a TIFF file using fabio.

        Parameters
        ----------
        file_path : Path
            Path to the TIFF file.
        average_frames : bool, optional
            Whether to average frames if the file contains multiple frames.
            Default is False.
        return_frames : bool, optional
            If True, return a list of frames and metadata instead of a single
            data array. Default is False.
        invert_mask : bool, optional
            Whether to invert the mask values. Not applicable here.
            Default is False.

        Returns
        -------
        tuple
            Data and metadata as described above.
        """
        try:
            with fabio.open(str(file_path)) as tiff_image:
                n_frames = getattr(tiff_image, "nframes", 1)
                data_list = []
                metadata_list = []

                for i in range(n_frames):
                    frame = tiff_image.getframe(i) if n_frames > 1 else tiff_image
                    frame_data = frame.data.astype(np.float32)
                    data_list.append(frame_data)

                    frame_metadata = frame.header if self.load_metadata else {}
                    metadata_list.append(frame_metadata)

                if average_frames and n_frames > 1:
                    averaged_data = np.mean(data_list, axis=0)
                    self.logger.debug(f"Averaged {n_frames} frames from '{file_path}'.")
                    return averaged_data, metadata_list[:1]

                if return_frames:
                    self.logger.debug(f"Loaded {n_frames} frames from '{file_path}'.")
                    return data_list, metadata_list

                if n_frames > 0:
                    self.logger.debug(f"Loaded first frame from '{file_path}'.")
                    return data_list[0], metadata_list[0]
                else:
                    self.logger.warning(f"No frames found in '{file_path}'.")
                    return None, None

        except Exception as e:
            self.logger.error(f"Error loading TIFF file '{file_path}': {e}")
            return None, None

    def load_edf_file(
        self,
        file_path: Path,
        average_frames: bool = False,
        return_frames: bool = False,
        invert_mask: bool = False,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load data from an EDF file using fabio.

        Parameters
        ----------
        file_path : Path
            Path to the EDF file.
        average_frames : bool, optional
            Whether to average frames if the file contains multiple frames.
            Default is False.
        return_frames : bool, optional
            If True, return a list of frames and metadata instead of a single
            data array. Default is False.
        invert_mask : bool, optional
            Whether to invert the mask values. Use this when loading mask files.
            Default is False.

        Returns
        -------
        tuple
            Data and metadata as described above.
        """
        try:
            with fabio.open(str(file_path)) as edf_image:
                n_frames = getattr(edf_image, "nframes", 1)
                data_list = []
                metadata_list = []

                for i in range(n_frames):
                    frame = edf_image.getframe(i) if n_frames > 1 else edf_image
                    frame_data = frame.data.astype(np.float32)
                    data_list.append(frame_data)

                    frame_metadata = frame.header if self.load_metadata else {}
                    metadata_list.append(frame_metadata)

                if invert_mask:
                    self.logger.info(f"Inverting mask values for '{file_path}'.")
                    data_list = [
                        np.where(frame == 1, 0, 1).astype(float) for frame in data_list
                    ]

                if average_frames:
                    averaged_data = np.mean(data_list, axis=0)
                    self.logger.debug(f"Averaged {n_frames} frames from '{file_path}'.")
                    return averaged_data, metadata_list[:1]

                if return_frames:
                    self.logger.debug(f"Loaded {n_frames} frames from '{file_path}'.")
                    return data_list, metadata_list

                if n_frames > 0:
                    self.logger.debug(f"Loaded first frame from '{file_path}'.")
                    return data_list[0], metadata_list[0]
                else:
                    self.logger.warning(f"No frames found in '{file_path}'.")
                    return None, None

        except Exception as e:
            self.logger.error(f"Error loading EDF file '{file_path}': {e}")
            return None, None

    def load_poni_file(
        self,
        file_path: Path,
        average_frames: bool = False,
        return_frames: bool = False,
        invert_mask: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Load data from a PONI file using pyFAI.

        Parameters
        ----------
        file_path : Path
            Path to the PONI file.
        average_frames : bool, optional
            Not applicable for PONI files.
            Default is False.
        return_frames : bool, optional
            Not applicable for PONI files.
            Default is False.
        invert_mask : bool, optional
            Not applicable for PONI files.
            Default is False.

        Returns
        -------
        tuple
            For PONI files, return an empty NumPy array for data and a dictionary
            containing the integrator metadata.
        """
        try:
            # Load the azimuthal integrator (pyFAI)
            integrator = pyFAI.load(str(file_path))

            # Convert integrator's parameters to dictionary metadata
            integrator_metadata = {
                "poni_file": str(file_path),
                "distance": integrator.dist,
                "poni1": integrator.poni1,
                "poni2": integrator.poni2,
                "rot1": integrator.rot1,
                "rot2": integrator.rot2,
                "rot3": integrator.rot3,
                "splineFile": integrator.splineFile,
                "detector": str(integrator.detector),
            }

            self.logger.debug(f"Loaded PONI file '{file_path}' as metadata.")

            # Return an empty array as data and integrator metadata
            return np.array([]), integrator_metadata

        except Exception as e:
            self.logger.error(f"Error loading PONI file '{file_path}': {e}")
            return None, None

    def load_instprm_file(
        self,
        file_path: Path,
        average_frames: bool = False,
        return_frames: bool = False,
        invert_mask: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], None]:
        """
        Load data from an INSTPRM file.

        Parameters
        ----------
        file_path : Path
            Path to the INSTPRM file.
        average_frames : bool, optional
            Not applicable for INSTPRM files.
            Default is False.
        return_frames : bool, optional
            Not applicable for INSTPRM files.
            Default is False.
        invert_mask : bool, optional
            Not applicable for INSTPRM files.
            Default is False.

        Returns
        -------
        tuple
            Metadata dictionary and None for data.
        """
        try:
            instprm_data: Dict[str, Any] = {}
            with open(file_path, "r") as f:
                for line in f:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip()
                        try:
                            instprm_data[key] = float(value)
                        except ValueError:
                            instprm_data[key] = value
            self.logger.debug(f"Loaded INSTPRM file '{file_path}'.")
            return instprm_data, None

        except Exception as e:
            self.logger.error(f"Error loading INSTPRM file '{file_path}': {e}")
            return None, None

    def load_json_file(
        self,
        file_path: Path,
        average_frames: bool = False,
        return_frames: bool = False,
        invert_mask: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], None]:
        """
        Load data from a JSON file.

        Parameters
        ----------
        file_path : Path
            Path to the JSON file.
        average_frames : bool, optional
            Not applicable for JSON files.
            Default is False.
        return_frames : bool, optional
            Not applicable for JSON files.
            Default is False.
        invert_mask : bool, optional
            Not applicable for JSON files.
            Default is False.

        Returns
        -------
        tuple
            Metadata dictionary and None for data.
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            self.logger.debug(f"Loaded JSON file '{file_path}'.")
            return data, None

        except Exception as e:
            self.logger.error(f"Error loading JSON file '{file_path}': {e}")
            return None, None

    # New loader methods

    def load_h5_file(
        self,
        file_path: Path,
        average_frames: bool = False,
        return_frames: bool = False,
        invert_mask: bool = False,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load data from an HDF5 (.h5) file using h5py.

        Parameters
        ----------
        file_path : Path
            Path to the HDF5 file.
        average_frames : bool, optional
            Whether to average frames if the file contains multiple frames.
            Default is False.
        return_frames : bool, optional
            If True, return a list of frames and metadata instead of a single
            data array. Default is False.
        invert_mask : bool, optional
            Not applicable for HDF5 files.
            Default is False.

        Returns
        -------
        tuple
            Data and metadata as described above.
        """
        try:
            with h5py.File(file_path, "r") as hdf_file:
                data_list = []
                metadata_list = []

                def visit_func(name, node):
                    if isinstance(node, h5py.Dataset):
                        dataset_data = node[()]
                        if isinstance(dataset_data, bytes):
                            # If dataset contains bytes (e.g., strings), store as metadata
                            metadata_list.append(
                                {name: dataset_data.decode("utf-8", "replace")}
                            )
                            # Add a placeholder empty array for data
                            data_list.append(np.array([]))
                        elif isinstance(dataset_data, np.ndarray):
                            if dataset_data.dtype.kind in ["u", "i", "f"]:
                                # Numeric data, convert to float32
                                dataset_data = dataset_data.astype(np.float32)
                                data_list.append(dataset_data)
                                # If metadata is to be loaded, store dataset's attributes
                                if self.load_metadata and node.attrs:
                                    frame_metadata = dict(node.attrs)
                                    metadata_list.append(frame_metadata)
                                else:
                                    metadata_list.append({})
                            else:
                                # Non-numeric data, store as metadata
                                metadata_list.append({name: dataset_data.tolist()})
                                data_list.append(np.array([]))
                        else:
                            # Non-numeric data, store as metadata
                            metadata_list.append({name: str(dataset_data)})
                            data_list.append(np.array([]))
                    else:
                        # If we find a group or link, ignore for now
                        pass

                hdf_file.visititems(visit_func)

                if not data_list:
                    self.logger.warning(f"No datasets found in '{file_path}'.")
                    return None, None

                # If average_frames is True, average all numeric datasets
                numeric_data_arrays = [d for d in data_list if d.size > 0]
                if average_frames and numeric_data_arrays:
                    averaged_data = np.mean(numeric_data_arrays, axis=0)
                    self.logger.debug(f"Averaged datasets from '{file_path}'.")
                    return averaged_data, metadata_list[:1] if metadata_list else None

                if return_frames:
                    self.logger.debug(f"Loaded datasets from '{file_path}'.")
                    return data_list, metadata_list

                # Return the first numeric dataset if available
                for d in data_list:
                    if d.size > 0:
                        return d, metadata_list[data_list.index(d)]
                # If no numeric dataset was found, return the first dataset's metadata
                return None, metadata_list[0] if metadata_list else None

        except Exception as e:
            self.logger.error(f"Error loading HDF5 file '{file_path}': {e}")
            return None, None

    def load_cbf_file(
        self,
        file_path: Path,
        average_frames: bool = False,
        return_frames: bool = False,
        invert_mask: bool = False,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load data from a CBF file using fabio.

        Parameters
        ----------
        file_path : Path
            Path to the CBF file.
        average_frames : bool, optional
            Whether to average frames if the file contains multiple frames.
            Default is False.
        return_frames : bool, optional
            If True, return a list of frames and metadata instead of a single
            data array. Default is False.
        invert_mask : bool, optional
            Whether to invert the mask values. Not applicable here.
            Default is False.

        Returns
        -------
        tuple
            Data and metadata as described above.
        """
        # CBF files are handled similarly to EDF and TIFF files
        return self.load_edf_file(
            file_path,
            average_frames=average_frames,
            return_frames=return_frames,
            invert_mask=invert_mask,
        )

    def load_mar3450_file(
        self,
        file_path: Path,
        average_frames: bool = False,
        return_frames: bool = False,
        invert_mask: bool = False,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load data from a MAR3450 file using fabio.

        Parameters
        ----------
        file_path : Path
            Path to the MAR3450 file.
        average_frames : bool, optional
            Whether to average frames if the file contains multiple frames.
            Default is False.
        return_frames : bool, optional
            If True, return a list of frames and metadata instead of a single
            data array. Default is False.
        invert_mask : bool, optional
            Whether to invert the mask values. Not applicable here.
            Default is False.

        Returns
        -------
        tuple
            Data and metadata as described above.
        """
        # MAR3450 files are handled similarly to EDF and TIFF files
        return self.load_edf_file(
            file_path,
            average_frames=average_frames,
            return_frames=return_frames,
            invert_mask=invert_mask,
        )

    def load_geom_file(
        self,
        file_path: Path,
        average_frames: bool = False,
        return_frames: bool = False,
        invert_mask: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], None]:
        """
        Load data from a .geom file.

        Parameters
        ----------
        file_path : Path
            Path to the .geom file.
        average_frames : bool, optional
            Not applicable for .geom files.
            Default is False.
        return_frames : bool, optional
            Not applicable for .geom files.
            Default is False.
        invert_mask : bool, optional
            Not applicable for .geom files.
            Default is False.

        Returns
        -------
        tuple
            Metadata dictionary and None for data.
        """
        try:
            with open(file_path, "r") as f:
                content = f.read()
            self.logger.debug(f"Loaded geom file '{file_path}'.")
            return {"content": content}, None
        except Exception as e:
            self.logger.error(f"Error loading geom file '{file_path}': {e}")
            return None, None

    def load_spline_file(
        self,
        file_path: Path,
        average_frames: bool = False,
        return_frames: bool = False,
        invert_mask: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], None]:
        """
        Load data from a .spline file.

        Parameters
        ----------
        file_path : Path
            Path to the .spline file.
        average_frames : bool, optional
            Not applicable for .spline files.
            Default is False.
        return_frames : bool, optional
            Not applicable for .spline files.
            Default is False.
        invert_mask : bool, optional
            Not applicable for .spline files.
            Default is False.

        Returns
        -------
        tuple
            Metadata dictionary and None for data.
        """
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
            self.logger.debug(f"Loaded spline file '{file_path}'.")
            return {"lines": lines}, None
        except Exception as e:
            self.logger.error(f"Error loading spline file '{file_path}': {e}")
            return None, None

    def load_png_file(
        self,
        file_path: Path,
        average_frames: bool = False,
        return_frames: bool = False,
        invert_mask: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[Any]]:
        """
        Load data from a PNG file using Pillow.

        Parameters
        ----------
        file_path : Path
            Path to the PNG file.
        average_frames : bool, optional
            Whether to average frames if the file contains multiple frames.
            Default is False.
        return_frames : bool, optional
            If True, return a list of frames and metadata instead of a single
            data array. Default is False.
        invert_mask : bool, optional
            Whether to invert the mask values. Use this when loading mask files.
            Default is False.

        Returns
        -------
        tuple
            Data array and metadata as described above.
        """
        try:
            with Image.open(file_path) as img:
                data = np.array(img)
                if invert_mask:
                    data = np.where(data == 1, 0, 1).astype(float)
                self.logger.debug(f"Loaded PNG file '{file_path}'.")
                return data.astype(np.float32), None
        except Exception as e:
            self.logger.error(f"Error loading PNG file '{file_path}': {e}")
            return None, None
