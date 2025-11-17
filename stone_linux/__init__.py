"""
stone-linux: PyTorch 2.10 with native SM 12.0 support for RTX 50-series GPUs

This package provides a convenient installer and utilities for PyTorch
compiled with native Blackwell architecture (SM 12.0) support.
"""

__version__ = "2.10.0a0"
__author__ = "PyTorch RTX Community"
__license__ = "BSD-3-Clause"

from stone_linux.verify import (
    verify_installation,
    check_cuda_available,
    get_gpu_info,
    check_compute_capability,
)
from stone_linux.installer import (
    install_pytorch_wheel,
    get_wheel_url,
    download_wheel,
)

__all__ = [
    "__version__",
    "verify_installation",
    "check_cuda_available",
    "get_gpu_info",
    "check_compute_capability",
    "install_pytorch_wheel",
    "get_wheel_url",
    "download_wheel",
]
