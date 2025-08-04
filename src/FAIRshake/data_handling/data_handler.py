# FAIRshake/data_handling/data_handler.py

import json
import logging
from typing import Any, Dict, Optional, List


class DataHandler:
    """
    A class containing utility functions for metadata handling.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def decode_metadata_str(self, metadata: Any) -> Optional[str]:
        """
        Decodes metadata from various formats to a UTF-8 string.

        Parameters
        ----------
        metadata : Any
            Metadata which can be of various types.

        Returns
        -------
        Optional[str]
            Decoded metadata string or None if decoding fails.
        """
        try:
            if isinstance(metadata, bytes):
                metadata_str = metadata.decode("utf-8")
            elif isinstance(metadata, str):
                metadata_str = metadata
            elif isinstance(metadata, (list, dict)):
                metadata_str = json.dumps(metadata)
            else:
                metadata_str = str(metadata)
            return metadata_str
        except Exception as exc:
            self.logger.error(f"Error decoding metadata: {exc}")
            return None

    def decode_and_parse_metadata(self, metadata: Any) -> Optional[Dict]:
        """
        Decodes metadata and parses it into a dictionary.

        Parameters
        ----------
        metadata : Any
            Metadata which can be of various types.

        Returns
        -------
        Optional[Dict]
            Parsed metadata dictionary or None if parsing fails.
        """
        try:
            metadata_str = self.decode_metadata_str(metadata)
            if metadata_str is None:
                return None
            metadata_dict = json.loads(metadata_str)
            return metadata_dict
        except json.JSONDecodeError as jde:
            self.logger.error(f"JSON decoding error: {jde}")
            return None
        except Exception as exc:
            self.logger.error(f"Unexpected error during metadata parsing: {exc}")
            return None

    def find_key_recursive(self, data: Any, target_key: str) -> Optional[Any]:
        """
        Recursively searches for a key in a nested dictionary or list.

        Parameters
        ----------
        data : Any
            The data structure to search.
        target_key : str
            The key to find.

        Returns
        -------
        Optional[Any]
            The value associated with the key, or None if not found.
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if key == target_key:
                    return value
                else:
                    result = self.find_key_recursive(value, target_key)
                    if result is not None:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = self.find_key_recursive(item, target_key)
                if result is not None:
                    return result
        return None

    def extract_naming_variables(
        self, metadata_dict: Dict, required_keys: List[str]
    ) -> Dict[str, Any]:
        """
        Extracts required keys from metadata for naming conventions.

        Parameters
        ----------
        metadata_dict : Dict
            The metadata dictionary.
        required_keys : List[str]
            List of keys to extract.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the extracted keys and their values.
        """
        naming_variables = {}
        for key in required_keys:
            value = self.find_key_recursive(metadata_dict, key)
            if value is None:
                self.logger.error(
                    f"Missing metadata key '{key}'. Using 'unknown' as default."
                )
                value = "unknown"
            naming_variables[key] = value
        return naming_variables

    def sanitize_metadata_keys(self, metadata: Any) -> Any:
        """
        Recursively sanitizes metadata keys and values by replacing spaces with underscores
        and converting bytes to strings.

        Parameters
        ----------
        metadata : Any
            The metadata to sanitize.

        Returns
        -------
        Any
            The sanitized metadata.
        """
        if isinstance(metadata, dict):
            sanitized_dict = {}
            for key, value in metadata.items():
                # Decode key if it's bytes
                if isinstance(key, bytes):
                    try:
                        key = key.decode("utf-8")
                    except UnicodeDecodeError:
                        key = key.decode("utf-8", errors="replace")
                # Replace spaces in key
                if isinstance(key, str):
                    key = key.replace(" ", "_")
                # Sanitize value
                sanitized_value = self.sanitize_metadata_keys(value)
                sanitized_dict[key] = sanitized_value
            return sanitized_dict
        elif isinstance(metadata, list):
            return [self.sanitize_metadata_keys(item) for item in metadata]
        elif isinstance(metadata, bytes):
            try:
                return metadata.decode("utf-8")
            except UnicodeDecodeError:
                return metadata.decode("utf-8", errors="replace")
        else:
            return metadata

    def flatten_dict(self, d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
        """
        Flattens a nested dictionary.

        Parameters
        ----------
        d : Dict
            The dictionary to flatten.
        parent_key : str, optional
            The base key string. Default is ''.
        sep : str, optional
            Separator between parent and child keys. Default is '_'.

        Returns
        -------
        Dict
            The flattened dictionary.
        """
        items = []
        for key, value in d.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(self.flatten_dict(value, new_key, sep=sep).items())
            elif isinstance(value, list):
                # Handle lists by enumerating indices
                for i, item in enumerate(value):
                    list_key = f"{new_key}{sep}{i}"
                    if isinstance(item, dict):
                        items.extend(self.flatten_dict(item, list_key, sep=sep).items())
                    else:
                        items.append((list_key, item))
            else:
                items.append((new_key, value))
        return dict(items)
