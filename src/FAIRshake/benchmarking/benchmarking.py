#!/usr/bin/env python
# coding: utf-8

"""
Benchmarking Module for FAIRshake
---------------------------------

This module benchmarks the FAIRshake pipeline by measuring the performance of the combined pipeline run processes.
It prepares the filesystem structure, verifies the existence and correctness of necessary data files, processes all
datasets (skipping incomplete ones), and then executes the pipeline, measuring performance for different batch sizes.

Author: Finley
Date: 12/11/2024
License: MIT
"""

import logging
import shutil
import time
import json
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import matplotlib.pyplot as plt
import threading
import hashlib
import urllib.parse

from FAIRshake.execution_pipeline.pipeline import ExecutionPipeline
from FAIRshake.utils.logger import setup_logging
from FAIRshake.utils.resource_utils import get_max_workers, get_system_resources
import fabio

# Suppress pyFAI INFO messages by adjusting its logger level
logging.getLogger("pyFAI").setLevel(logging.WARNING)

download_lock = threading.Lock()

# Load dataset information from file_hashes.json
with open(Path(__file__).parent / "file_hashes.json", "r") as f:
    file_hashes = json.load(f)

URL_BASE = file_hashes["url_base"]
DATASETS = [
    {
        "name": dataset["name"],
        "poni": dataset["filename"] if dataset["type"] == "poni" else None,
        "image": dataset["filename"] if dataset["type"] == "image" else None,
        "size": dataset["size"],
        "hash": dataset["hash"],
        "megapixels": dataset.get("megapixels"),
    }
    for dataset in file_hashes["datasets"]
]

# Merge datasets with the same name and keep separate hashes and sizes for 'image' and 'poni' files
merged_datasets = {}
for dataset in DATASETS:
    name = dataset["name"]
    if name not in merged_datasets:
        merged_datasets[name] = {"name": name}
    if dataset["poni"]:
        merged_datasets[name]["poni"] = dataset["poni"]
        merged_datasets[name]["poni_hash"] = dataset["hash"]
        merged_datasets[name]["poni_size"] = dataset["size"]
    if dataset["image"]:
        merged_datasets[name]["image"] = dataset["image"]
        merged_datasets[name]["image_hash"] = dataset["hash"]
        merged_datasets[name]["image_size"] = dataset["size"]
    if "megapixels" in dataset:
        merged_datasets[name]["megapixels"] = dataset["megapixels"]
DATASETS = list(merged_datasets.values())


def verify_file_hash(file_path: Path, expected_hash: str) -> bool:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_hash


def benchmark(
    data_dir: Optional[str] = None,
    cleanup_input: bool = True,
    cleanup_output: bool = True,
    iterations: int = 5,
    batch_sizes: Optional[List[int]] = None,
    files_per_dataset: int = 10,
    log_level: str = "ERROR",
) -> Dict[str, Any]:
    """
    Runs the full benchmarking workflow.

    Parameters:
        data_dir (Optional[str]): Path to the data directory. Uses a temporary directory if None.
        cleanup_input (bool): Whether to clean up the input data directory after benchmarking.
        cleanup_output (bool): Whether to clean up the exported data directory after benchmarking.
        iterations (int): Number of iterations to run the pipeline for each dataset and batch size.
        batch_sizes (Optional[List[int]]): List of batch sizes to benchmark.
        files_per_dataset (int): Number of image files per dataset.
        log_level (str): Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").

    Returns:
        Dict[str, Any]: A dictionary containing benchmarking results.
    """
    # Prepare directories
    data_dir = _prepare_directories(data_dir)
    input_dir = data_dir / "input"
    export_dir = data_dir / "export"
    results_dir = data_dir / "results"
    log_dir = data_dir / "log"

    for directory in [input_dir, export_dir, results_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger, _, _ = setup_logging(log_dir=log_dir, log_level=log_level)

    logger.info("Starting FAIRshake Benchmarking Module.")
    logger.info(f"Using base data directory: {data_dir}")

    # Set default batch sizes if not provided
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16]

    # Prepare filesystem: download files and create duplicates
    _prepare_filesystem(logger, input_dir, files_per_dataset)

    # Run benchmarking for each batch size
    all_results = {}
    for batch_size in batch_sizes:
        logger.info(f"Benchmarking with batch size {batch_size}...")
        all_results[batch_size] = _run_benchmarking(
            logger, input_dir, export_dir, batch_size, iterations, cleanup_output
        )

    # Generate plots and save results
    _generate_plots(all_results, logger, results_dir, iterations, files_per_dataset)

    # Save results to JSON
    _save_results(all_results, logger, results_dir, iterations, files_per_dataset)

    # Cleanup directories based on parameters
    _cleanup_data_dirs(
        logger,
        input_dir,
        export_dir,
        log_dir,
        results_dir,
        cleanup_input,
        cleanup_output,
    )

    logger.info("Benchmarking completed successfully.")
    return all_results


def _prepare_directories(data_dir: Optional[str] = None) -> Path:
    """
    Prepare the main data directory. If not specified, use a temporary directory.
    """
    if data_dir is None:
        data_dir = Path(tempfile.gettempdir()) / "fairshake_benchmark_data"
    else:
        data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _prepare_filesystem(
    logger: logging.Logger, input_dir: Path, files_per_dataset: int
):
    logger.info(
        "Preparing filesystem: Downloading files and creating duplicates as needed."
    )

    # Loop through each dataset and download files
    for dataset in DATASETS:
        name = dataset["name"]
        dataset_dir = input_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Download and verify the image file
        if "image" in dataset and dataset["image"]:
            # Sanitize the file name
            sanitized_image_name = urllib.parse.quote(dataset["image"])
            image_url = urllib.parse.urljoin(URL_BASE + "/", sanitized_image_name)
            image_path = dataset_dir / dataset["image"]
            if not image_path.exists():
                logger.info(
                    f"Downloading image file {dataset['image']} for dataset {name}"
                )
                response = requests.get(image_url)
                response.raise_for_status()
                with open(image_path, "wb") as f:
                    f.write(response.content)
            # Verify the hash of the downloaded image file
            expected_hash = dataset.get("image_hash")
            if expected_hash and not verify_file_hash(image_path, expected_hash):
                logger.error(f"Hash mismatch for image file {dataset['image']}")
                raise ValueError(f"Hash mismatch for image file {dataset['image']}")
            else:
                logger.info(f"Image file {dataset['image']} verified successfully")

            # Create duplicates if needed
            for i in range(1, files_per_dataset):
                duplicate_path = (
                    dataset_dir / f"{image_path.stem}_{i}{image_path.suffix}"
                )
                if not duplicate_path.exists():
                    shutil.copy(image_path, duplicate_path)

        # Download and verify the poni file
        if "poni" in dataset and dataset["poni"]:
            # Sanitize the file name
            sanitized_poni_name = urllib.parse.quote(dataset["poni"])
            poni_url = urllib.parse.urljoin(URL_BASE + "/", sanitized_poni_name)
            poni_path = dataset_dir / dataset["poni"]
            if not poni_path.exists():
                logger.info(
                    f"Downloading poni file {dataset['poni']} for dataset {name}"
                )
                response = requests.get(poni_url)
                response.raise_for_status()
                with open(poni_path, "wb") as f:
                    f.write(response.content)
            # Verify the hash of the downloaded poni file
            expected_hash = dataset.get("poni_hash")
            if expected_hash and not verify_file_hash(poni_path, expected_hash):
                logger.error(f"Hash mismatch for poni file {dataset['poni']}")
                raise ValueError(f"Hash mismatch for poni file {dataset['poni']}")
            else:
                logger.info(f"Poni file {dataset['poni']} verified successfully")


def _verify_or_download_file(
    logger: logging.Logger, dest_dir: Path, file_name: str, attempts: int = 3
) -> bool:
    """
    Verify if a file exists and is valid. If invalid or does not exist, attempt to re-download the file.
    """
    dest_file = dest_dir / file_name
    if not dest_file.exists():
        return _download_file(logger, dest_dir, file_name, attempts=attempts)

    if dest_file.stat().st_size == 0:
        logger.warning(f"File {dest_file} is empty. Attempting to re-download.")
        return _download_file(logger, dest_dir, file_name, attempts=attempts)

    logger.debug(f"File {dest_file} verified successfully.")
    return True


def _download_file(
    logger: logging.Logger, dest_dir: Path, file_name: str, attempts: int = 3
) -> bool:
    with download_lock:
        """
        Download a file from URL_BASE to the destination directory with retry logic.
        """
        url = f"{URL_BASE}/{file_name}"
        dest_file = dest_dir / file_name
        logger.debug(f"Attempting to download {file_name} from {url}...")

        for attempt in range(1, attempts + 1):
            try:
                response = requests.get(url, timeout=30, stream=True)
                response.raise_for_status()

                with open(dest_file, "wb") as f:
                    shutil.copyfileobj(response.raw, f)

                content_length = response.headers.get("Content-Length")
                if content_length and dest_file.stat().st_size != int(content_length):
                    logger.warning(f"File {file_name} downloaded but size mismatch.")
                    if attempt < attempts:
                        time.sleep(5)
                        continue
                    else:
                        logger.error(
                            f"Failed to fully download {file_name} after {attempts} attempts."
                        )
                        return False

                logger.debug(f"Downloaded {file_name} successfully.")
                return True

            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt} failed to download {file_name}: {e}")
                if attempt < attempts:
                    time.sleep(5)
                else:
                    logger.error(
                        f"Failed to download {file_name} after {attempts} attempts."
                    )
                    return False

        return False


def _copy_file(logger: logging.Logger, src: Path, dest: Path):
    """
    Copies a single file from src to dest, preserving metadata.
    """
    try:
        if (
            not dest.exists()
            or dest.stat().st_size == 0
            or src.stat().st_mtime > dest.stat().st_mtime
        ):
            shutil.copy2(src, dest)
            logger.debug(f"Copied {src.name} to {dest}")
        else:
            logger.debug(f"Destination file {dest} already exists and is up to date.")
    except Exception as e:
        logger.error(f"Failed to copy {src} to {dest}: {e}")


def _run_benchmarking(
    logger: logging.Logger,
    input_dir: Path,
    export_dir: Path,
    batch_size: int,
    iterations: int,
    cleanup_output: bool,
) -> Dict[str, Dict[str, Any]]:
    """
    Runs the benchmarking workflow for the combined pipeline with the specified batch size.
    """
    results = {}
    for ds in DATASETS:
        dataset_dir = input_dir / ds["name"]
        if not dataset_dir.exists():
            logger.warning(f"Dataset directory {dataset_dir} does not exist. Skipping.")
            continue

        # Check if necessary files are present and not corrupted
        poni_file = dataset_dir / ds["poni"]
        if not poni_file.exists() or poni_file.stat().st_size == 0:
            logger.warning(
                f"Required .poni file {poni_file} is missing or empty. Skipping dataset {ds['name']}."
            )
            continue

        image_files = sorted(
            dataset_dir.glob(f"{ds['image'].split('.')[0]}*{Path(ds['image']).suffix}")
        )
        image_files = [f for f in image_files if f.stat().st_size > 0]

        if not image_files:
            logger.warning(
                f"No valid image files found for dataset {ds['name']}. Skipping."
            )
            continue

        logger.info(f"Processing dataset: {ds['name']}")
        try:
            dataset_results = _process_dataset(
                logger,
                ds,
                dataset_dir,
                export_dir,
                batch_size,
                iterations,
                cleanup_output,
                image_files,
            )
            results[ds["name"]] = dataset_results
        except Exception as e:
            logger.error(f"Error processing dataset {ds['name']}: {e}")
    return results


def _process_dataset(
    logger: logging.Logger,
    ds: Dict[str, str],
    dataset_dir: Path,
    export_dir: Path,
    batch_size: int,
    iterations: int,
    cleanup_output: bool,
    image_files: List[Path],
) -> Dict[str, Any]:
    """
    Process a single dataset by running multiple iterations of the pipeline.
    """
    poni_file = dataset_dir / ds["poni"]
    main_image_path = dataset_dir / ds["image"]
    megapixels = _calculate_megapixels(logger, main_image_path)

    iteration_times = []
    iteration_fps = []
    for i in range(iterations):
        iteration_time = _run_pipeline_iteration(
            logger,
            ds["name"],
            dataset_dir,
            export_dir,
            poni_file,
            image_files,
            batch_size,
            i,
            cleanup_output,
        )
        iteration_times.append(iteration_time)
        if iteration_time > 0:
            frames_processed = len(image_files)
            fps = frames_processed / iteration_time
            iteration_fps.append(fps)
        else:
            iteration_fps.append(0.0)

    valid_times = [t for t in iteration_times if t > 0]
    if not valid_times:
        combined_fps = "N/A"
    else:
        avg_time = sum(valid_times) / len(valid_times)
        frames_processed = len(image_files)
        combined_fps = round(frames_processed / avg_time, 2) if avg_time else "N/A"

    return {
        "combined": combined_fps,
        "megapixels": round(megapixels, 2) if megapixels else 0.0,
        "iterations": {
            "iteration_times": iteration_times,
            "iteration_fps": [round(f, 2) for f in iteration_fps],
        },
    }


def _calculate_megapixels(logger: logging.Logger, image_path: Path) -> float:
    """
    Calculate the megapixels for the main image file of the dataset.
    """
    if not image_path.exists() or image_path.stat().st_size == 0:
        logger.warning(f"Main image file {image_path} not found or is empty.")
        return 0.0
    try:
        fabio_image = fabio.open(str(image_path))
        image_data = fabio_image.data
        if image_data is None or image_data.size == 0:
            raise ValueError("Image data is empty or unreadable.")
        return (image_data.shape[0] * image_data.shape[1]) / 1e6
    except Exception as e:
        logger.warning(f"Could not read image file {image_path}: {e}")
        return 0.0


def _run_pipeline_iteration(
    logger: logging.Logger,
    dataset_name: str,
    dataset_dir: Path,
    export_dir: Path,
    poni_file: Path,
    image_files: List[Path],
    batch_size: int,
    iteration_index: int,
    cleanup_output: bool,
) -> float:
    """
    Runs a single iteration of the pipeline for the given dataset.
    """
    # Create directories
    batch_dir = export_dir / dataset_name / f"batch_size_{batch_size}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    iteration_dir = batch_dir / f"iteration_{iteration_index}"
    iteration_dir.mkdir(parents=True, exist_ok=True)

    naming_convention = f"{dataset_name}_batch{batch_size}"

    # Initialize the ExecutionPipeline with enable_file_logging=False
    pipeline = _initialize_pipeline(
        logger,
        dataset_dir,
        iteration_dir,
        poni_file,
        image_files,
        batch_size,
        naming_convention,
        enable_file_logging=False,
    )

    start_time = time.perf_counter()
    try:
        pipeline.run()
        # Check if pipeline exported the results
        export_files = list(iteration_dir.glob(f"{naming_convention}_*.fxye"))
        if not export_files:
            logger.warning(
                f"No exported files found for dataset {dataset_name}, "
                f"batch size {batch_size}, iteration {iteration_index}."
            )
        else:
            exported_file_names = [f.name for f in export_files]
            logger.info(
                f"Exported files for dataset {dataset_name}, batch size {batch_size}, "
                f"iteration {iteration_index}: {exported_file_names}"
            )
    except Exception as e:
        logger.error(
            f"Error running pipeline for dataset {dataset_name}, "
            f"batch size {batch_size}, iteration {iteration_index}: {e}"
        )
        return 0.0
    finally:
        end_time = time.perf_counter()
        iteration_time = end_time - start_time
        logger.debug(
            f"Iteration {iteration_index} for dataset {dataset_name}, "
            f"batch size {batch_size} took {iteration_time:.2f} seconds."
        )
        # Conditionally clean up the export files
        if cleanup_output:
            for f in iteration_dir.glob("*"):
                try:
                    if f.is_file():
                        f.unlink()
                    elif f.is_dir():
                        shutil.rmtree(f)
                    logger.debug(f"Deleted exported file/directory: {f}")
                except Exception as e:
                    logger.error(f"Failed to delete exported file/directory {f}: {e}")
        else:
            logger.debug(
                f"Retained exported files for dataset {dataset_name}, "
                f"batch size {batch_size}, iteration {iteration_index}."
            )
        return iteration_time


def _initialize_pipeline(
    logger: logging.Logger,
    dataset_dir: Path,
    iteration_dir: Path,
    poni_file: Path,
    image_files: List[Path],
    batch_size: int,
    naming_convention: str,
    enable_file_logging: bool = True,
) -> ExecutionPipeline:
    """
    Initializes and returns an ExecutionPipeline configured for a specific dataset iteration.
    """
    data_file_types = list({img.suffix.lower() for img in image_files})
    metadata_file_types = {poni_file.suffix.lower()}

    pipeline_params = {
        "input_base_dir": str(dataset_dir),
        "output_base_dir": str(iteration_dir),
        "batch_size": batch_size,
        "data_file_types": data_file_types,
        "metadata_file_types": list(metadata_file_types),
        "require_metadata": False,
        "load_metadata_files": False,
        "load_detector_metadata": False,
        "require_all_formats": False,
        "average_frames": False,
        "enable_profiling": False,
        "tf_data_debug_mode": False,
        "pattern": "**/*",
        "preprocessing_config": {},
        "enable_preprocessing": False,
        "enable_integration": True,
        "integration_config": {
            "poni_file_path": str(poni_file),
            "npt_radial": 500,
            "unit": "2th_deg",
            "do_solid_angle": False,
            "error_model": "poisson",
            "radial_range": None,
            "azimuth_range": None,
            "polarization_factor": 0.99,
            "method": ("full", "histogram", "cython"),
            "safe": True,
        },
        "enable_exporting": True,
        "exporting_config": {
            "output_directory": str(iteration_dir),
            "naming_convention": f"{naming_convention}_{{iter}}",
            "options": {"do_remove_nan": True},
            "file_format": "fxye",
        },
        "enable_file_logging": enable_file_logging,
    }
    logger.debug(
        f"Initialized pipeline with naming convention: "
        f"{pipeline_params['exporting_config']['naming_convention']} and batch size {batch_size}"
    )
    return ExecutionPipeline(**pipeline_params)


def _generate_plots(
    all_results: Dict[int, Dict[str, Dict[str, Any]]],
    logger: logging.Logger,
    results_dir: Path,
    iterations: int,
    files_per_dataset: int,
):
    """
    Generates and saves a consolidated plot for all batch sizes, showing Pipeline Flow Rate (FPS) vs. Image Size (Megapixels).
    """
    resources = get_system_resources()
    hardware_info = "\n".join([f"{key}: {value}" for key, value in resources.items()])

    logger.info(
        "Generating benchmarking plots focusing on combined pipeline flow rate."
    )
    if not all_results:
        logger.error("No benchmarking results available to plot.")
        return

    plt.figure(figsize=(16, 10))
    marker_style = "o"
    colors = plt.cm.get_cmap("tab10")

    dataset_megapixels = {}
    for batch_size, datasets in all_results.items():
        for dataset_name, result in datasets.items():
            if dataset_name not in dataset_megapixels:
                megapixels = result.get("megapixels", 0.0)
                dataset_megapixels[dataset_name] = megapixels

    sorted_datasets = sorted(dataset_megapixels.items(), key=lambda x: x[1])

    max_fps = 0
    for batch_size, datasets in all_results.items():
        for result in datasets.values():
            combined_fps = result.get("combined", 0.0)
            if isinstance(combined_fps, (int, float)):
                max_fps = max(max_fps, combined_fps)

    batch_sizes = sorted(all_results.keys())
    color_map = {batch_size: colors(i) for i, batch_size in enumerate(batch_sizes)}

    for batch_size in batch_sizes:
        datasets = all_results[batch_size]
        batch_fps = []
        batch_megapixels = []
        for dataset_name, result in datasets.items():
            combined_fps = result.get("combined", 0.0)
            megapixels = dataset_megapixels.get(dataset_name, 0.0)
            if combined_fps == "N/A" or combined_fps == 0.0 or megapixels == 0.0:
                logger.debug(
                    f"Skipping dataset {dataset_name} for batch size {batch_size} due to invalid data."
                )
                continue
            batch_fps.append(float(combined_fps))
            batch_megapixels.append(megapixels)

        if not batch_fps:
            logger.warning(
                f"No valid data points for batch size {batch_size}. Skipping."
            )
            continue

        plt.scatter(
            batch_megapixels,
            batch_fps,
            marker=marker_style,
            color=color_map[batch_size],
            label=f"Batch Size {batch_size}",
            s=100,
            alpha=0.7,
        )

    y_min, y_max = plt.ylim()
    for dataset_name, megapixels in sorted_datasets:
        if megapixels == 0.0:
            logger.debug(
                f"Skipping dataset {dataset_name} for vertical line due to invalid megapixels."
            )
            continue
        plt.axvline(x=megapixels, linestyle="--", color="black", alpha=0.6, linewidth=1)
        plt.text(
            megapixels - (max(dataset_megapixels.values()) * 0.02),
            y_min + (y_max - y_min) * 0.05,  # Position 5% up from the bottom
            dataset_name,
            fontsize=10,
            rotation=90,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"
            ),
        )

    plt.xlabel("Image Size (Megapixels)", fontsize=14)
    plt.ylabel("Pipeline Flow Rate (FPS)", fontsize=14)
    plt.title("FAIRshake Pipeline Benchmarking", fontsize=16)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color_map[bs], label=f"Batch Size {bs}") for bs in batch_sizes
    ]
    legend = plt.legend(
        handles=legend_elements,
        title="Batch Sizes",
        loc="upper right",
    )

    combined_info = f"Hardware Info:\n{hardware_info}\n\nRun Info:\nFrames per Dataset: {files_per_dataset}\nIterations: {iterations}"

    plt.text(
        0.95,
        0.85,  # Move it below the legend to avoid overlap
        combined_info,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="none"
        ),
    )

    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.75, 1])

    plot_path = results_dir / "benchmark_pipeline_flow_rate_vs_megapixels.png"
    try:
        plt.savefig(plot_path, bbox_inches="tight")
        logger.info(
            f"Saved consolidated pipeline flow rate vs. megapixels plot to {plot_path}"
        )
    except Exception as e:
        logger.error(f"Failed to save plot to {plot_path}: {e}")
    finally:
        plt.close()


def _save_results(
    all_results: Dict[int, Dict[str, Dict[str, Any]]],
    logger: logging.Logger,
    results_dir: Path,
    iterations: int,
    files_per_dataset: int,
):
    """
    Saves the benchmarking results to a JSON file.
    """
    # Collect additional data
    resources = get_system_resources()
    hardware_info = {key: value for key, value in resources.items()}

    # Include additional data in all_results
    all_results["metadata"] = {
        "hardware_info": hardware_info,
        "iterations": iterations,
        "files_per_dataset": files_per_dataset,
    }

    # Collect dataset megapixels
    dataset_megapixels = {}
    for batch_size, datasets in all_results.items():
        if batch_size == "metadata":
            continue
        for dataset_name, result in datasets.items():
            if dataset_name not in dataset_megapixels:
                megapixels = result.get("megapixels", 0.0)
                dataset_megapixels[dataset_name] = megapixels

    all_results["metadata"]["dataset_megapixels"] = dataset_megapixels

    results_path = results_dir / "benchmark_results.json"
    try:
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=4)
        logger.info(f"Saved benchmarking results to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save benchmarking results to {results_path}: {e}")

def _cleanup_data_dirs(
    logger: logging.Logger,
    input_dir: Path,
    export_dir: Path,
    log_dir: Path,
    results_dir: Path,
    cleanup_input: bool,
    cleanup_output: bool,
):
    """
    Cleans up the input and export directories based on 'cleanup_input' and 'cleanup_output' parameters.
    """
    if cleanup_input and input_dir.exists():
        shutil.rmtree(input_dir)
        logger.info(f"Cleaned up input data directory: {input_dir}")
    else:
        logger.info(f"Retaining input data directory: {input_dir}")
    if cleanup_output and export_dir.exists():
        shutil.rmtree(export_dir)
        logger.info(f"Cleaned up export data directory: {export_dir}")
    else:
        logger.info(f"Retaining export data directory: {export_dir}")

    logger.info(f"Retaining log directory: {log_dir}")
    logger.info(f"Retaining results directory: {results_dir}")
