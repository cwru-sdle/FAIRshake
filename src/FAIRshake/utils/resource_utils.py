import os
import multiprocessing
import platform
import logging
import subprocess
from typing import Dict, Optional

# Import psutil at the top
try:
    import psutil
except ImportError:
    psutil = None


def get_allocated_cores() -> int:
    """
    Determine the number of allocated CPU cores using various methods
    (e.g., os.sched_getaffinity, cgroup quotas, HPC environment variables).
    If all methods fail, defaults to the total number of CPU cores.

    Returns:
        int: The number of allocated CPU cores.
    """
    logger = logging.getLogger(__name__)

    # Determine the total number of CPU cores
    try:
        total_cores = multiprocessing.cpu_count()
    except NotImplementedError:
        logger.warning(
            "Unable to determine total CPU cores. Defaulting allocated cores to 1."
        )
        return 1

    # Attempt using os.sched_getaffinity (mainly on Linux)
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = os.sched_getaffinity(0)
            allocated_cores = len(affinity)
            if allocated_cores > 0:
                logger.info(
                    f"Allocated cores from os.sched_getaffinity: {allocated_cores}"
                )
                return allocated_cores
            else:
                logger.warning("os.sched_getaffinity returned an empty set.")
        except Exception as e:
            logger.warning(f"Error using os.sched_getaffinity: {e}")

    # Attempt using cgroup CPU quota settings (for Linux systems)
    try:
        cpu_quota_path = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
        cpu_period_path = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
        if os.path.exists(cpu_quota_path) and os.path.exists(cpu_period_path):
            with (
                open(cpu_quota_path, "r") as f_quota,
                open(cpu_period_path, "r") as f_period,
            ):
                quota_us = int(f_quota.read().strip())
                period_us = int(f_period.read().strip())
                if quota_us > 0 and period_us > 0:
                    allocated_cores = quota_us // period_us
                    if allocated_cores > 0:
                        logger.info(
                            f"Allocated cores from cgroup CPU quotas: {allocated_cores}"
                        )
                        return allocated_cores
                else:
                    logger.warning("Invalid cgroup CPU quota or period values.")
    except Exception as e:
        logger.warning(f"Error reading cgroup CPU quotas: {e}")

    # Check HPC environment variables
    env_vars = [
        "SLURM_CPUS_ON_NODE",
        "SLURM_CPUS_PER_TASK",
        "SLURM_JOB_CPUS_PER_NODE",
        "SLURM_TASKS_PER_NODE",
        "SLURM_NTASKS",
        "PBS_NUM_PPN",
        "PBS_NP",
    ]
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            try:
                allocated = int(value.split("(")[0].split("x")[0])
                logger.info(f"Allocated cores from {var}: {allocated}")
                return allocated
            except ValueError:
                logger.warning(
                    f"Environment variable {var} is set but could not parse integer from '{value}'."
                )

    # Fallback to total_cores if no method worked
    logger.info(f"No specific allocation found. Using total cores: {total_cores}")
    return total_cores


def get_max_workers() -> int:
    """
    Calculate the maximum number of worker threads to use.
    It uses the number of allocated cores minus one, ensuring at least one worker.

    Returns:
        int: The number of worker threads to use.
    """
    logger = logging.getLogger(__name__)
    allocated_cores = get_allocated_cores()
    # Reserve one core for system tasks
    max_workers = max(1, allocated_cores - 1)
    logger.info(f"Max workers: {max_workers} (Allocated cores: {allocated_cores})")
    return max_workers


def get_system_resources() -> Dict[str, Optional[str]]:
    """
    Collect system resource information including CPU, GPU, and memory.

    Returns:
    -------
    Dict[str, Optional[str]]:
        A dictionary containing system resource details.
    """
    logger = logging.getLogger(__name__)
    resources = {}

    # CPU Cores (allocated)
    allocated_cores = get_allocated_cores()
    resources["CPU Cores"] = str(allocated_cores)

    # CPU Frequency
    resources["CPU Frequency"] = "Unavailable"
    try:
        if psutil:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq and cpu_freq.max:
                resources["CPU Frequency"] = f"{cpu_freq.max:.2f} MHz"
                logger.info(f"CPU Frequency from psutil: {resources['CPU Frequency']}")
            else:
                logger.warning(
                    "psutil.cpu_freq() returned None or missing 'max' attribute."
                )
        elif "arm" in platform.machine().lower():
            # On Apple Silicon or other ARM architectures, attempt to get CPU frequency using sysctl
            try:
                freq_output = subprocess.check_output(
                    ["sysctl", "-n", "hw.cpufrequency_max"], text=True
                ).strip()
                # Some systems might return frequency in Hz, others in different units
                try:
                    freq_hz = int(freq_output)
                    cpu_freq_mhz = freq_hz / 1e6  # Convert Hz to MHz
                    resources["CPU Frequency"] = f"{cpu_freq_mhz:.2f} MHz"
                    logger.info(
                        f"CPU Frequency from sysctl: {resources['CPU Frequency']}"
                    )
                except ValueError:
                    logger.warning(
                        f"sysctl output for CPU frequency is invalid: '{freq_output}'"
                    )
            except subprocess.CalledProcessError as e:
                logger.warning(f"sysctl command failed with error: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error when using sysctl: {e}")
        else:
            logger.warning(
                "psutil not available and not on ARM architecture. Cannot retrieve CPU frequency."
            )
    except Exception as e:
        logger.warning(f"Error detecting CPU frequency: {e}")

    # Total Memory
    resources["Total Memory"] = "Unavailable"
    try:
        if psutil:
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            resources["Total Memory"] = f"{total_memory_gb:.2f} GB"
            logger.info(f"Total Memory from psutil: {resources['Total Memory']}")
        else:
            logger.warning("psutil not installed. Memory info unavailable.")
    except Exception as e:
        logger.warning(f"Error detecting total memory: {e}")

    # GPU detection
    resources["GPU"] = "No GPU detected"
    resources["GPU Memory"] = "N/A"

    try:
        if "arm" in platform.machine().lower():
            # Attempt to detect Apple Metal performance shading (MPS)
            try:
                import torch

                if torch.backends.mps.is_available():
                    resources["GPU"] = "Apple Silicon GPU (MPS)"
                    resources["GPU Memory"] = "Shared with system memory (Unavailable)"
                    logger.info("Apple Silicon GPU detected with MPS support.")
                else:
                    resources["GPU"] = "Apple Silicon GPU (MPS not available)"
                    logger.info("Apple Silicon GPU detected but MPS is not available.")
            except ImportError:
                logger.warning("torch not installed. GPU info unavailable.")
            except Exception as e:
                logger.warning(f"Error detecting GPU on Apple Silicon: {e}")
        else:
            # Check for NVIDIA GPUs using nvidia-smi
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.total",
                        "--format=csv,noheader",
                    ],
                    text=True,
                    capture_output=True,
                    check=True,
                )
                lines = result.stdout.strip().split("\n")
                if lines:
                    # Assuming single GPU for simplicity; can be expanded for multiple GPUs
                    gpu_info = lines[0].split(",")
                    gpu_name = gpu_info[0].strip()
                    gpu_memory_mb = gpu_info[1].strip().replace(" MB", "")
                    resources["GPU"] = gpu_name
                    if gpu_memory_mb.isdigit():
                        resources["GPU Memory"] = f"{gpu_memory_mb} MB"
                    else:
                        resources["GPU Memory"] = "Unknown"
                    logger.info(
                        f"GPU detected: {resources['GPU']} with {resources['GPU Memory']} memory."
                    )
            except subprocess.CalledProcessError as e:
                logger.warning(f"nvidia-smi command failed with error: {e}")
            except FileNotFoundError:
                # nvidia-smi not found, likely no NVIDIA GPU
                resources["GPU"] = "No NVIDIA GPUs detected"
                logger.info("nvidia-smi not found. No NVIDIA GPU detected.")
            except Exception as e:
                logger.warning(f"Error detecting GPU with nvidia-smi: {e}")
    except Exception as e:
        logger.warning(f"Error during GPU detection: {e}")

    return resources
