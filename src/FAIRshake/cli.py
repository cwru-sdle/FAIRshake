# FAIRshake/cli.py

import click
import logging
import json
from pathlib import Path

from FAIRshake.execution_pipeline.pipeline import ExecutionPipeline
from FAIRshake.benchmarking.benchmarking import (
    benchmark,
)  # If you have a benchmarking module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """FAIRshake Command Line Interface"""
    pass


@cli.command()
@click.option(
    "--input-base-dir",
    type=click.Path(exists=True),
    required=True,
    help="Input base directory containing data files.",
)
@click.option(
    "--output-base-dir",
    type=click.Path(),
    required=True,
    help="Output base directory for processed data.",
)
@click.option("--batch-size", default=32, type=int, help="Batch size for processing.")
@click.option(
    "--data-file-types",
    default=".ge2 .tif .edf .cbf .mar3450 .h5 .png",
    help="Space-separated list of data file extensions to include.",
)
@click.option(
    "--metadata-file-types",
    default=".json .poni .instprm .geom .spline",
    help="Space-separated list of metadata file extensions to include.",
)
@click.option(
    "--require-metadata/--no-require-metadata",
    default=True,
    help="Require metadata files for image files.",
)
@click.option(
    "--load-metadata-files/--no-load-metadata-files",
    default=True,
    help="Load additional metadata files.",
)
@click.option(
    "--load-detector-metadata/--no-load-detector-metadata",
    default=False,
    help="Load detector-specific metadata.",
)
@click.option(
    "--require-all-formats/--no-require-all-formats",
    default=False,
    help="Require all metadata file types to be present.",
)
@click.option(
    "--average-frames/--no-average-frames",
    default=False,
    help="Average multiple frames into a single frame.",
)
@click.option(
    "--enable-profiling/--no-enable-profiling",
    default=False,
    help="Enable TensorFlow profiling.",
)
@click.option(
    "--tf-data-debug-mode/--no-tf-data-debug-mode",
    default=False,
    help="Enable TensorFlow data debug mode.",
)
@click.option("--pattern", default="**/*", help="Glob pattern to match files.")
@click.option(
    "--log-dir", default=None, type=click.Path(), help="Directory for storing logs."
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Logging level.",
)
@click.option(
    "--enable-preprocessing/--no-enable-preprocessing",
    default=True,
    help="Enable preprocessing.",
)
@click.option(
    "--preprocessing-config",
    type=click.Path(exists=True),
    default=None,
    help="Path to JSON file with preprocessing configuration.",
)
@click.option(
    "--enable-integration/--no-enable-integration",
    default=True,
    help="Enable integration.",
)
@click.option(
    "--integration-config",
    type=click.Path(exists=True),
    default=None,
    help="Path to JSON file with integration configuration.",
)
@click.option(
    "--enable-exporting/--no-enable-exporting", default=True, help="Enable exporting."
)
@click.option(
    "--exporting-config",
    type=click.Path(exists=True),
    default=None,
    help="Path to JSON file with exporting configuration.",
)
def run_pipeline(**kwargs):
    """Run the FAIRshake data processing pipeline."""
    # Convert data_file_types and metadata_file_types from space-separated strings to lists
    data_file_types = kwargs["data_file_types"].split()
    metadata_file_types = kwargs["metadata_file_types"].split()

    # Read configurations from JSON files if provided
    preprocessing_config = None
    if kwargs["preprocessing_config"]:
        with open(kwargs["preprocessing_config"], "r") as f:
            preprocessing_config = json.load(f)

    integration_config = None
    if kwargs["integration_config"]:
        with open(kwargs["integration_config"], "r") as f:
            integration_config = json.load(f)

    exporting_config = None
    if kwargs["exporting_config"]:
        with open(kwargs["exporting_config"], "r") as f:
            exporting_config = json.load(f)

    # Prepare pipeline parameters
    pipeline_params = {
        "input_base_dir": kwargs["input_base_dir"],
        "output_base_dir": kwargs["output_base_dir"],
        "batch_size": kwargs["batch_size"],
        "data_file_types": data_file_types,
        "metadata_file_types": metadata_file_types,
        "require_metadata": kwargs["require_metadata"],
        "load_metadata_files": kwargs["load_metadata_files"],
        "load_detector_metadata": kwargs["load_detector_metadata"],
        "require_all_formats": kwargs["require_all_formats"],
        "average_frames": kwargs["average_frames"],
        "enable_profiling": kwargs["enable_profiling"],
        "tf_data_debug_mode": kwargs["tf_data_debug_mode"],
        "pattern": kwargs["pattern"],
        "log_dir": kwargs["log_dir"],
        "log_level": kwargs["log_level"],
        "enable_preprocessing": kwargs["enable_preprocessing"],
        "preprocessing_config": preprocessing_config,
        "enable_integration": kwargs["enable_integration"],
        "integration_config": integration_config,
        "enable_exporting": kwargs["enable_exporting"],
        "exporting_config": exporting_config,
    }

    # Initialize and run the pipeline
    pipeline = ExecutionPipeline(**pipeline_params)
    pipeline.run()
    click.echo("Pipeline execution completed.")


@cli.command()
@click.option("--data-dir", default=None, help="Path to the data directory.")
@click.option(
    "--cleanup-input/--no-cleanup-input",
    default=True,
    help="Clean up input data after benchmarking.",
)
@click.option(
    "--cleanup-output/--no-cleanup-output",
    default=True,
    help="Clean up output data after benchmarking.",
)
@click.option("--iterations", default=5, type=int, help="Number of iterations to run.")
@click.option("--batch-size", multiple=True, type=int, help="Batch sizes to benchmark.")
@click.option(
    "--files-per-dataset", default=10, type=int, help="Number of files per dataset."
)
@click.option(
    "--log-level",
    default="ERROR",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Logging level.",
)
def benchmark_cmd(
    data_dir,
    cleanup_input,
    cleanup_output,
    iterations,
    batch_size,
    files_per_dataset,
    log_level,
):
    """Run the benchmarking workflow."""
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, log_level))

    logger.info("Starting benchmark...")

    result = benchmark(
        data_dir=data_dir,
        cleanup_input=cleanup_input,
        cleanup_output=cleanup_output,
        iterations=iterations,
        batch_sizes=list(batch_size) if batch_size else None,
        files_per_dataset=files_per_dataset,
        log_level=log_level,
    )
    click.echo("Benchmarking completed.")
    # Optionally, output results
    click.echo(result)
