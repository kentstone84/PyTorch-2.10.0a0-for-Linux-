"""Configuration for stone-linux package."""

import sys
from typing import Dict, List

# Package version
VERSION = "2.10.0a0"

# GitHub repository
REPO_OWNER = "kentstone84"
REPO_NAME = "PyTorch-2.10.0a0-for-Linux-"
REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}"

# Release information
RELEASE_TAG = "v2.10.0a0"
RELEASE_URL = f"{REPO_URL}/releases/tag/{RELEASE_TAG}"

# Wheel URLs by Python version
WHEEL_URLS: Dict[str, str] = {
    "3.10": f"{REPO_URL}/releases/download/{RELEASE_TAG}/torch-2.10.0a0-cp310-cp310-linux_x86_64.whl",
    "3.11": f"{REPO_URL}/releases/download/{RELEASE_TAG}/torch-2.10.0a0-cp311-cp311-linux_x86_64.whl",
    "3.12": f"{REPO_URL}/releases/download/{RELEASE_TAG}/torch-2.10.0a0-cp312-cp312-linux_x86_64.whl",
    "3.13": f"{REPO_URL}/releases/download/{RELEASE_TAG}/torch-2.10.0a0-cp313-cp313-linux_x86_64.whl",
    "3.14": f"{REPO_URL}/releases/download/{RELEASE_TAG}/torch-2.10.0a0-cp314-cp314-linux_x86_64.whl",
}

# Supported Python versions
SUPPORTED_PYTHON_VERSIONS: List[str] = ["3.10", "3.11", "3.12", "3.13", "3.14"]

# Minimum driver version for RTX 50-series
MIN_DRIVER_VERSION = "570.00"

# Expected compute capability for RTX 50-series
EXPECTED_COMPUTE_CAPABILITY = (12, 0)

# Supported GPUs
SUPPORTED_GPUS = [
    "NVIDIA GeForce RTX 5090",
    "NVIDIA GeForce RTX 5080",
    "NVIDIA GeForce RTX 5070 Ti",
    "NVIDIA GeForce RTX 5070",
]

# CUDA versions
REQUIRED_CUDA_VERSION = "13.0"
MIN_CUDA_VERSION = "12.0"


def get_python_version() -> str:
    """Get the current Python version as a string (e.g., '3.12')."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def is_python_version_supported() -> bool:
    """Check if the current Python version is supported."""
    return get_python_version() in SUPPORTED_PYTHON_VERSIONS


def get_wheel_url_for_current_python() -> str:
    """Get the wheel URL for the current Python version."""
    python_version = get_python_version()
    if python_version not in WHEEL_URLS:
        raise ValueError(
            f"Python {python_version} is not supported. "
            f"Supported versions: {', '.join(SUPPORTED_PYTHON_VERSIONS)}"
        )
    return WHEEL_URLS[python_version]
