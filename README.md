# FAIRshake

FAIRshake (Sample Handling and Analysis Kit for Experiments) is a comprehensive data processing pipeline designed for efficient benchmarking and processing of datasets, particularly in diffraction data analysis. It includes modules for benchmarking, data loading, preprocessing, integration, and exporting.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [From Source](#from-source)
- [Usage](#usage)
  - [Command-Line Interface](#command-line-interface)
  - [Data Processing Pipeline](#data-processing-pipeline)
  - [Benchmarking](#benchmarking)
  - [Processing Raw Data with Jupyter Notebooks](#processing-raw-data-with-jupyter-notebooks)
  - [Computing File Hashes](#computing-file-hashes)
  - [Programmatic Usage](#programmatic-usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Contact Information](#contact-information)

## Features

- **Benchmarking Modules**: Assess the performance of data processing workflows.
- **Data Loading**: Efficient handling of large-scale datasets.
- **Preprocessing**: Data cleaning, normalization, and noise reduction.
- **Integration**: Combine data from various formats and sources seamlessly.
- **Exporting**: Output processed data in multiple formats for further analysis.

## Installation

### Requirements

- Python 3.11 or higher

### From Source

Clone the repository and install FAIRshake locally:

```bash
git clone https://github.com/FinleyHolt/SHAKE.git
cd SHAKE
pip install .

# FAIRshake

FAIRshake (Sample Handling and Analysis Kit for Experiments) is a comprehensive data processing pipeline designed for efficient benchmarking and processing of datasets, particularly in diffraction data analysis. It includes modules for benchmarking, data loading, preprocessing, integration, and exporting.

## Features

- **Benchmarking Modules**: Assess the performance of data processing workflows.
- **Data Loading**: Efficient handling of large-scale datasets.
- **Preprocessing**: Data cleaning, normalization, and noise reduction.
- **Integration**: Combine data from various formats and sources seamlessly.
- **Exporting**: Output processed data in multiple formats for further analysis.

## Installation

### Requirements

- Python 3.11 or higher

### From Source

Clone the repository and install FAIRshake locally:

```bash
git clone https://github.com/FinleyHolt/SHAKE.git
cd SHAKE
pip install .
```

## Usage

FAIRshake provides command-line tools and modules for data processing, benchmarking, and integration of diffraction data.

### Command-Line Interface

After installation, you can use the fairshake command. Use fairshake --help to see available commands:

```bash
fairshake --help
```

#### Data Processing Pipeline

To run the data processing pipeline on your dataset:

```bash
fairshake process --config <config-file> --data-dir <data-directory> --output-dir <output-directory>
```
##### Example Configuration File

Create a configuration file (e.g., config.json) specifying parameters for preprocessing, integration, and exporting:

```json
{
  "preprocessing": {
    "dark_field_path": "path/to/dark_field.ge2",
    "mask_file_path": "path/to/mask.edf",
    "invert_mask": true,
    "min_intensity": 0.0,
    "max_intensity": null
  },
  "integration": {
    "poni_file_path": "calibration_files/det0.poni",
    "npt_radial": 500,
    "unit": "2th_deg",
    "do_solid_angle": false,
    "error_model": "poisson",
    "radial_range": [3, 13],
    "azimuth_range": [-180, 180],
    "polarization_factor": 0.99,
    "method": ["full", "histogram", "cython"]
  },
  "exporting": {
    "output_directory": "path/to/output",
    "naming_convention": "{GE_filenumber}_{iter}",
    "options": {
      "do_remove_nan": true,
      "unit": "2th_deg"
    },
    "file_format": "fxye"
  }
}
```

#### Benchmarking

To benchmark the performance of the data processing pipeline:

```bash
fairshake benchmark --data-dir <data-directory> \
                    --iterations <iterations> \
                    --batch-size <batch-size> \
                    --files-per-dataset <files-per-dataset>
```

Example:

```bash
fairshake benchmark --data-dir data/benchmark_files \
                    --iterations 1 \
                    --batch-size 5 \
                    --files-per-dataset 10
```

### Programmatic Usage

You can use FAIRshake modules directly in your Python scripts:

```python
from FAIRshake.execution_pipeline.pipeline import ExecutionPipeline

# Configuration Parameters
input_base_dir = 'path/to/input'
output_base_dir = 'path/to/output'

# Preprocessing configuration
preprocessing_config = {
    "dark_field_path": "path/to/dark_field.ge2",
    "mask_file_path": "path/to/mask.edf",
    "invert_mask": True,
    "min_intensity": 0.0,
    "max_intensity": None,
}

# Integration configuration
integration_config = {
    "poni_file_path": "calibration_files/det0.poni",
    "npt_radial": 500,
    "unit": "2th_deg",
    "do_solid_angle": False,
    "error_model": "poisson",
    "radial_range": (3, 13),
    "azimuth_range": [-180, 180],
    "polarization_factor": 0.99,
    "method": ["full", "histogram", "cython"]
}

# Exporting configuration
exporting_config = {
    "output_directory": output_base_dir,
    "naming_convention": "{GE_filenumber}_{iter}",
    "options": {
        "do_remove_nan": True,
        "unit": "2th_deg"
    },
    "file_format": "fxye"
}

# Pipeline parameters
pipeline_params = {
    "input_base_dir": input_base_dir,
    "output_base_dir": output_base_dir,
    "batch_size": 10,
    "data_file_types": ['.ge2', '.tif', '.edf', '.cbf', '.mar3450', '.h5', '.png'],
    "metadata_file_types": ['.json', '.poni', '.instprm', '.geom', '.spline'],
    "require_metadata": True,
    "load_metadata_files": True,
    "load_detector_metadata": False,
    "require_all_formats": False,
    "average_frames": False,
    "enable_profiling": True,
    "tf_data_debug_mode": False,
    "pattern": '*/*/*',
    "preprocessing_config": preprocessing_config,
    "enable_preprocessing": True,
    "enable_integration": True,
    "integration_config": integration_config,
    "enable_exporting": True,
    "exporting_config": exporting_config,
    "log_level": "ERROR"
}

# Initialize the Execution Pipeline
pipeline = ExecutionPipeline(**pipeline_params)

# Run the Pipeline
pipeline.run()
```

Ensure that you define preprocessing_config, integration_config, and exporting_config according to your requirements.

## Help and Support

For detailed usage and options, use the help command:

```bash
fairshake process --help
fairshake benchmark --help
```

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

### Steps to Contribute

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE.txt](LICENSE.txt) file for details.

## Contact Information

For support or inquiries:

- **Author**: Finley Holt
- **Email**: finley0454@gmail.com
- **GitHub**: [FinleyHolt](https://github.com/FinleyHolt)