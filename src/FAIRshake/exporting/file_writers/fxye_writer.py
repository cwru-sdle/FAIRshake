# FAIRshake/exporting/file_writers/fxye_writer.py

import logging
from pathlib import Path
from typing import Any

import numpy as np


class FxyeWriter:
    """
    A class for writing data to FXYE format files.
    """

    def __init__(self, output_directory: Path, logger: logging.Logger):
        """
        Initialize the FxyeWriter.

        Parameters
        ----------
        output_directory : Path
            Directory where FXYE files will be saved.
        logger : logging.Logger
            Logger instance for logging messages.
        """
        self.output_directory: Path = output_directory
        self.logger: logging.Logger = logger

    def write(self, x: np.ndarray, y: np.ndarray, e: np.ndarray, filename: str) -> None:
        """
        Write the spectra data to an FXYE file.

        Parameters
        ----------
        x : np.ndarray
            Radial positions.
        y : np.ndarray
            Intensities.
        e : np.ndarray
            Errors.
        filename : str
            Name of the output file without extension.
        """
        safe_filename: str = self._sanitize_filename(filename)
        file_path: Path = self.output_directory / f"{safe_filename}.fxye"

        # Multiply x values by 100 to convert to centidegrees
        x = x * 100

        # Calculate radial step
        if len(x) > 1:
            radial_step: float = x[1] - x[0]
        else:
            radial_step = 0.0

        # Create header
        header: str = (
            f"{file_path}\n"
            f"BANK 1 {len(x)} {len(y)} CONS 3.0 {radial_step:.18e} 0 0 FXYE"
        )

        # Write data to file
        try:
            with open(file_path, "w") as f:
                f.write(header + "\n")
                for xi, yi, ei in zip(x, y, e):
                    f.write(f"{xi:.18e}\t{yi:.18e}\t{ei:.18e}\n")
            self.logger.debug(f"Successfully wrote FXYE file: {file_path}")
        except Exception as exc:
            self.logger.error(
                f"Error writing FXYE file {file_path}: {exc}", exc_info=True
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
