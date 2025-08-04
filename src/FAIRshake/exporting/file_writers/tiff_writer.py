# FAIRshake/exporting/file_writers/tiff_writer.py

import logging
from pathlib import Path
from typing import Any

import numpy as np
import imageio  # Ensure imageio is installed: pip install imageio


class TiffWriter:
    """
    A class for writing 2D data to TIFF format files.
    """

    def __init__(self, output_directory: Path, logger: logging.Logger):
        """
        Initialize the TiffWriter.

        Parameters
        ----------
        output_directory : Path
            Directory where TIFF files will be saved.
        logger : logging.Logger
            Logger instance for logging messages.
        """
        self.output_directory: Path = output_directory
        self.logger: logging.Logger = logger

    def write(self, data: np.ndarray, filename: str) -> None:
        """
        Write the 2D data to a TIFF file.

        Parameters
        ----------
        data : np.ndarray
            2D array data to be saved as TIFF.
        filename : str
            Name of the output file without extension.
        """
        safe_filename: str = self._sanitize_filename(filename)
        file_path: Path = self.output_directory / f"{safe_filename}.tiff"

        # Write data to file
        try:
            imageio.imwrite(file_path, data.astype(np.float32))
            self.logger.debug(f"Successfully wrote TIFF file: {file_path}")
        except Exception as exc:
            self.logger.error(
                f"Error writing TIFF file {file_path}: {exc}", exc_info=True
            )

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize the filename by replacing or removing invalid characters.

        Parameters
        ----------
        filename : str
            The original filename with potential invalid characters.

        Returns
        -------
        str
            A sanitized filename safe for filesystem usage.
        """
        invalid_chars: str = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")
        return filename.strip()
